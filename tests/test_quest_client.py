"""The XRoboToolkit Quest client's stream, and the offline path that replays it.

The packets come from tools/quest_sim_client.py, which follows the Quest
client's source (v1.0.1). These tests pin what the server makes of them.
"""

from __future__ import annotations

import json
import signal
import socket
import struct
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot_teleoperator_rebot_vr.config_rebot_vr import DEFAULT_BASE_T_ANCHOR
from lerobot_teleoperator_rebot_vr.hardware.simulated import SimulatedFollower
from lerobot_teleoperator_rebot_vr.runtime.cli import build_parser, validate_args
from lerobot_teleoperator_rebot_vr.tools import quest_sim_client as quest
from lerobot_teleoperator_rebot_vr.tools.print_vr_data import GripOffset
from lerobot_teleoperator_rebot_vr.vr.pose_mapping import RelativePoseMapper, TeleopState
from lerobot_teleoperator_rebot_vr.vr.xr_v1 import (
    CMD_FUNCTION,
    PacketStreamDecoder,
    V1TrackingSource,
)

XR_TO_BASE = np.asarray(DEFAULT_BASE_T_ANCHOR, dtype=np.float64)[:3, :3]
NS = 1_000_000_000


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _feed(source: V1TrackingSource, right: quest.ControllerState, *, at_ns: int, **tracking):
    packet = quest.tracking_packet(
        quest.tracking_object(quest.REST_LEFT, right, time_stamp_ns=at_ns, **tracking)
    )
    source.feed_bytes(packet, received_monotonic_ns=at_ns)
    return source.latest_sample()


# --- wire format -------------------------------------------------------------


def test_client_packet_has_the_quest_layout() -> None:
    packet = quest.pack_client_packet(0x6D, b"abc", timestamp_ms=0x0102030405060708)

    assert packet[0] == 0x3F and packet[1] == 0x6D and packet[-1] == 0xA5
    assert struct.unpack_from("<i", packet, 2)[0] == 3
    assert packet[6:9] == b"abc"
    assert struct.unpack_from("<q", packet, 9)[0] == 0x0102030405060708
    assert len(packet) == 15 + 3

    (decoded,) = PacketStreamDecoder().feed(packet)
    assert decoded["cmd"] == CMD_FUNCTION and decoded["direction"] == "SEND"


def test_handshake_needs_no_reply_and_identifies_the_client() -> None:
    source = V1TrackingSource()
    for packet in quest.handshake_packets():
        source.feed_bytes(packet)
    source.feed_bytes(quest.heartbeat_packet())

    stats = source.stats()
    assert stats.packet_count == 3
    assert stats.tracking_frame_count == 0 and stats.invalid_frame_count == 0
    assert stats.client_version == "TestDevice|1.0|1.0.1"
    assert source.latest_sample() is None


def test_tracking_object_has_the_keys_the_quest_client_sends() -> None:
    tracking = quest.tracking_object(quest.REST_LEFT, quest.REST_RIGHT, time_stamp_ns=5)

    assert list(tracking) == ["predictTime", "appState", "Controller", "timeStampNs", "Input"]
    assert list(tracking["Controller"]["right"]) == [
        "axisX", "axisY", "axisClick", "grip", "trigger",
        "primaryButton", "secondaryButton", "menuButton", "pose",
    ]
    # The Tracking object travels as a JSON string inside the function message.
    body = json.loads(PacketStreamDecoder().feed(quest.tracking_packet(tracking))[0]["body_str"])
    assert body["functionName"] == "Tracking" and isinstance(body["value"], str)
    assert json.loads(body["value"]) == tracking


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.0, "0"), (1.0, "1"), (-0.5, "-0.5"), (0.1, "0.1"), (1e-6, "1E-06"), (-2.5e-7, "-2.5E-07")],
)
def test_floats_print_like_csharp_round_trip_format(value: float, text: str) -> None:
    assert quest.format_single(value) == text
    assert float(text) == pytest.approx(float(np.float32(value)))


def test_quest_sample_parses_with_buttons_and_no_status_field() -> None:
    source = V1TrackingSource(side="right")
    state = replace(
        quest.REST_RIGHT, grip=0.75, trigger=0.25, primary_button=True, menu_button=True
    )

    sample = _feed(source, state, at_ns=7 * NS)

    assert sample is not None and sample.side == "right"
    assert sample.grip == pytest.approx(0.75) and sample.trigger == pytest.approx(0.25)
    assert sample.primary_button is True and sample.secondary_button is False
    assert sample.status is None  # the Quest client sends no per-controller status
    assert sample.tracking_timestamp_ns == 7 * NS
    assert source.stats().invalid_frame_count == 0


def test_controller_toggle_off_is_an_invalid_frame_not_a_stale_pose() -> None:
    source = V1TrackingSource()
    tracking = quest.tracking_object(quest.REST_LEFT, quest.REST_RIGHT, time_stamp_ns=1)
    del tracking["Controller"]  # what the client sends with only "Send" switched on

    source.feed_bytes(quest.tracking_packet(tracking))

    assert source.latest_sample() is None
    assert source.stats().invalid_frame_count == 1


# --- frame ---------------------------------------------------------------------


def test_wire_pose_is_right_handed_x_right_y_up_z_back() -> None:
    # Unity: X right, Y up, Z forward. On the wire forward must be -Z.
    position, quaternion = quest.unity_to_wire_pose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(position, [0.1, 0.2, -0.3])
    np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), np.eye(3), atol=1e-12)

    # Turning right in Unity is +angle about +Y (left-handed). Right-handed,
    # about +Y up, turning right is a negative angle.
    unity_turn_right = Rotation.from_rotvec([0.0, np.deg2rad(30.0), 0.0]).as_quat()
    _, wire = quest.unity_to_wire_pose((0, 0, 0), unity_turn_right)
    np.testing.assert_allclose(
        Rotation.from_quat(wire).as_rotvec(), [0.0, -np.deg2rad(30.0), 0.0], atol=1e-12
    )


def _mapped_motion(unity_offset=(0.0, 0.0, 0.0), unity_rotvec_deg=(0.0, 0.0, 0.0)):
    """Robot-frame target change for a Unity-frame controller motion with Grip held."""
    source = V1TrackingSource(side="right")
    mapper = RelativePoseMapper(
        side="right", xr_to_world=XR_TO_BASE, grip_press_threshold=0.6, grip_release_threshold=0.4
    )
    ee_position, ee_rotation = np.array([0.3, 0.0, 0.3]), np.eye(3)
    start = quest.REST_RIGHT

    def step(state, at_ns):
        sample = _feed(source, state, at_ns=at_ns)
        return mapper.update(sample, ee_position, ee_rotation, now_ns=at_ns)

    step(start, 1 * NS)  # Grip released: arms the clutch
    assert step(replace(start, grip=1.0), 2 * NS).state is TeleopState.ACTIVE
    moved = replace(
        start,
        grip=1.0,
        position=tuple(np.asarray(start.position) + np.asarray(unity_offset)),
        quaternion_xyzw=tuple(Rotation.from_rotvec(np.deg2rad(unity_rotvec_deg)).as_quat()),
    )
    update = step(moved, 3 * NS)
    assert update.state is TeleopState.ACTIVE and update.target is not None
    turn = Rotation.from_matrix(update.target.rotation @ ee_rotation.T).as_rotvec()
    return update.target.position - ee_position, np.rad2deg(turn)


@pytest.mark.parametrize(
    ("unity_offset", "robot_offset"),
    [
        ((0.0, 0.0, 0.1), (0.1, 0.0, 0.0)),  # hand forward -> arm forward (+X)
        ((0.1, 0.0, 0.0), (0.0, -0.1, 0.0)),  # hand right -> arm right (-Y)
        ((0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),  # hand up -> arm up (+Z)
    ],
)
def test_quest_translation_maps_to_the_robot_axes(unity_offset, robot_offset) -> None:
    offset, turn = _mapped_motion(unity_offset=unity_offset)
    np.testing.assert_allclose(offset, robot_offset, atol=1e-6)
    np.testing.assert_allclose(turn, 0.0, atol=1e-4)


@pytest.mark.parametrize(
    ("unity_rotvec_deg", "robot_rotvec_deg"),
    [
        ((0.0, 30.0, 0.0), (0.0, 0.0, -30.0)),  # turn right -> negative about Z up
        ((30.0, 0.0, 0.0), (0.0, 30.0, 0.0)),  # nose down -> positive about Y left
        ((0.0, 0.0, 30.0), (-30.0, 0.0, 0.0)),  # roll left (counter-clockwise from behind) -> negative about X forward
    ],
)
def test_quest_rotation_maps_to_the_robot_axes(unity_rotvec_deg, robot_rotvec_deg) -> None:
    offset, turn = _mapped_motion(unity_rotvec_deg=unity_rotvec_deg)
    np.testing.assert_allclose(offset, 0.0, atol=1e-6)
    np.testing.assert_allclose(turn, robot_rotvec_deg, atol=1e-4)


def test_print_tool_offset_reads_forward_left_up() -> None:
    offset = GripOffset(press=0.6)
    held = {"is_tracking": True, "squeeze": 1.0, "grip_quat": [0, 0, 0, 1]}

    assert "Grip not held" in offset.update({**held, "squeeze": 0.0, "grip_pos": [0, 0, 0]})
    offset.update({**held, "grip_pos": [1.0, 2.0, 3.0]})
    line = offset.update({**held, "grip_pos": [1.1, 1.8, 3.05]})

    assert "forward=+0.100 m" in line and "left=-0.200 m" in line and "up=+0.050 m" in line


# --- pose-jump guard -----------------------------------------------------------


def _jump_mapper(**options) -> tuple[V1TrackingSource, RelativePoseMapper]:
    mapper = RelativePoseMapper(
        side="right", xr_to_world=XR_TO_BASE, grip_press_threshold=0.6,
        grip_release_threshold=0.4, stale_timeout_s=10.0, **options,
    )
    return V1TrackingSource(side="right"), mapper


def _play(source, mapper, states, *, step_ns=NS // 90):
    ee_position, ee_rotation = np.array([0.3, 0.0, 0.3]), np.eye(3)
    updates = []
    for index, state in enumerate(states, start=1):
        sample = _feed(source, state, at_ns=index * step_ns)
        updates.append(mapper.update(sample, ee_position, ee_rotation, now_ns=index * step_ns))
    return updates


def test_pose_jump_is_followed_when_the_guard_is_off() -> None:
    source, mapper = _jump_mapper()
    held = replace(quest.REST_RIGHT, grip=1.0)

    updates = _play(source, mapper, [quest.REST_RIGHT, held, replace(held, position=(0, 0, 0))])

    assert updates[-1].state is TeleopState.ACTIVE
    assert np.linalg.norm(updates[-1].target.position - [0.3, 0.0, 0.3]) > 1.0
    assert mapper.jump_count == 0


def test_pose_jump_releases_the_clutch_until_grip_is_released() -> None:
    source, mapper = _jump_mapper(jump_speed_m_s=2.0)
    held = replace(quest.REST_RIGHT, grip=1.0)
    lost = replace(held, position=(0.0, 0.0, 0.0))

    updates = _play(
        source, mapper,
        [quest.REST_RIGHT, held, lost, lost, replace(lost, grip=0.0), lost, lost],
    )

    states = [update.state for update in updates]
    assert states == [
        TeleopState.IDLE, TeleopState.ACTIVE,
        TeleopState.IDLE, TeleopState.IDLE,  # jump, then Grip still held
        TeleopState.IDLE,  # released
        TeleopState.ACTIVE, TeleopState.ACTIVE,  # pressed again: new reference
    ]
    assert updates[2].target is None and updates[2].require_release is True
    assert mapper.jump_count == 1
    np.testing.assert_allclose(updates[-1].target.position, [0.3, 0.0, 0.3], atol=1e-9)


def test_hand_speed_motion_does_not_trip_the_guard() -> None:
    source, mapper = _jump_mapper(jump_speed_m_s=2.0, jump_rate_rad_s=np.deg2rad(720.0))
    held = replace(quest.REST_RIGHT, grip=1.0)
    # 1.5 m/s and 360 deg/s at 90 frames a second.
    moving = [
        replace(
            held,
            position=(0.2 + 1.5 * index / 90, 1.1, 0.4),
            quaternion_xyzw=tuple(Rotation.from_rotvec([0, np.deg2rad(4.0 * index), 0]).as_quat()),
        )
        for index in range(1, 30)
    ]

    updates = _play(source, mapper, [quest.REST_RIGHT, held, *moving])

    assert all(update.state is TeleopState.ACTIVE for update in updates[1:])
    assert mapper.jump_count == 0


def test_rotation_jump_trips_the_guard() -> None:
    source, mapper = _jump_mapper(jump_rate_rad_s=np.deg2rad(720.0))
    held = replace(quest.REST_RIGHT, grip=1.0)
    flipped = replace(held, quaternion_xyzw=tuple(Rotation.from_rotvec([0, np.pi / 2, 0]).as_quat()))

    updates = _play(source, mapper, [quest.REST_RIGHT, held, flipped])

    assert updates[-1].state is TeleopState.IDLE and mapper.jump_count == 1


# --- dry run -------------------------------------------------------------------


def _args(*extra: str):
    args = build_parser().parse_args(["--motor-control-mode", "mit", *extra])
    validate_args(args)
    return args


def test_dry_run_is_off_by_default_and_is_dm_only() -> None:
    assert _args().dry_run is False
    assert _args().tracking_jump_speed_m_s == 0.0 and _args().tracking_jump_rate_deg_s == 0.0
    assert _args("--dry-run").dry_run is True
    with pytest.raises(ValueError, match="B601-DM"):
        _args("--dry-run", "--robot-model", "b601_rs", "--can-adapter", "socketcan",
              "--gripper-control-mode", "mit")


def test_simulated_follower_follows_commands_within_the_real_bounds() -> None:
    from lerobot.robots.rebot_b601_follower import RebotB601FollowerRobotConfig

    config = RebotB601FollowerRobotConfig(port="/not/opened", max_relative_target=5.0)
    follower = SimulatedFollower(config, start_deg={"shoulder_lift": -40.0})
    follower.connect(calibrate=True)

    assert follower.get_observation()["shoulder_lift.pos"] == pytest.approx(-40.0)
    sent = follower.send_action(
        {f"{name}.pos": (-90.0 if name == "shoulder_lift" else 0.0) for name in follower.motor_names}
    )
    assert sent["shoulder_lift.pos"] == pytest.approx(-45.0)  # the relative bound
    for step in range(1, 200):
        follower.advance(now_s=time.monotonic() + step * 0.01)
    assert follower.get_observation()["shoulder_lift.pos"] == pytest.approx(-45.0, abs=0.05)

    follower.disconnect()
    assert follower.bus is None and follower.motors == {}


class _Recorder:
    """Runtime observer: keeps what the loop measured and sent."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.steps: list[tuple[dict, dict | None, str]] = []

    def on_tracking(self, _sample, _tracking) -> None:
        return None

    def on_ready(self) -> None:
        self.ready.set()

    def on_step(self, observation, sent_action, status, _frame) -> None:
        self.steps.append((dict(observation), sent_action, status.state.value))

    def on_stop(self) -> None:
        return None


def _play_over_tcp(port: int, recorder: _Recorder, errors: list) -> None:
    try:
        deadline = time.monotonic() + 20.0
        while True:
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        with connection:
            for packet in quest.handshake_packets():
                connection.sendall(packet)
            rate = 90.0
            rest = quest.REST_RIGHT

            def send(state) -> None:
                tracking = quest.tracking_object(quest.REST_LEFT, state, time_stamp_ns=time.time_ns())
                connection.sendall(quest.tracking_packet(tracking))
                time.sleep(1.0 / rate)

            while not recorder.ready.wait(0.0):  # startup move: stream, Grip released
                send(rest)
            segments = [
                quest._hold(0.3, "released"),
                quest._ramp(0.2, "grip", "grip", 1.0),
                quest._hold(0.3, "held"),
                quest._translate(1.0, "forward", (0.0, 0.0, 0.10)),
                quest._hold(0.7, "settle"),
                quest._ramp(0.2, "release", "grip", 0.0),
                quest._hold(0.3, "released"),
            ]
            for _elapsed, _label, state in quest.scenario_states(segments, rate, rest):
                send(state)
    except Exception as exc:  # surfaced by the test
        errors.append(exc)


def test_dry_run_drives_the_whole_path_from_quest_packets(monkeypatch) -> None:
    """Quest bytes over TCP -> parse -> clutch -> IK -> MIT commands, no hardware."""
    from lerobot.robots.rebot_b601_follower import RebotB601Follower

    from lerobot_teleoperator_rebot_vr.ik.kinematics import B601Kinematics
    from lerobot_teleoperator_rebot_vr.runtime import real

    def no_hardware(*_args, **_kwargs):
        raise AssertionError("--dry-run must not construct or connect the real follower")

    monkeypatch.setattr(RebotB601Follower, "__init__", no_hardware)
    monkeypatch.setattr(RebotB601Follower, "connect", no_hardware)

    port = _free_port()
    args = _args(
        "--dry-run", "--port", str(port), "--duration", "5", "--status-rate", "0.2",
        "--robot-port", "/not/opened", "--log-level", "WARNING",
    )
    recorder = _Recorder()
    errors: list = []
    client = threading.Thread(target=_play_over_tcp, args=(port, recorder, errors), daemon=True)
    client.start()
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        real._run_arm(args, observer=recorder)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    client.join(timeout=5.0)

    assert not errors, errors
    states = [state for _obs, _sent, state in recorder.steps]
    assert "active" in states and states[-1] in ("idle", "stale")

    kinematics = B601Kinematics(None, "gripper_end")
    try:
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")

        def anchor(observation):
            q = np.deg2rad([observation[f"{name}.pos"] for name in names])
            return kinematics.wrist_anchor_pose(q)[0]

        first_active = states.index("active")
        last_active = len(states) - 1 - states[::-1].index("active")
        travel = anchor(recorder.steps[last_active][0]) - anchor(recorder.steps[first_active][0])
    finally:
        kinematics.close()
    # 10 cm forward at the package's unit scale, less what the loop has not caught up on.
    assert 0.06 < travel[0] < 0.11
    assert abs(travel[1]) < 0.01 and abs(travel[2]) < 0.01

    # Every command stayed inside the follower's limits and its relative bound.
    for observation, sent, _state in recorder.steps:
        if sent is None:
            continue
        for name in names:
            assert abs(sent[f"{name}.pos"] - observation[f"{name}.pos"]) <= args.max_relative_target_deg + 1e-6


# --- motor health ----------------------------------------------------------------


def _states(**codes):
    from types import SimpleNamespace

    return {name: SimpleNamespace(status_code=code) for name, code in codes.items()}


def test_motor_health_reports_faults_and_dropouts_but_not_startup() -> None:
    from lerobot_teleoperator_rebot_vr.runtime.motor_health import MotorHealthMonitor

    monitor = MotorHealthMonitor(dropout_reads=3)
    # Disabled before ever being seen enabled is startup, however long it lasts.
    for _ in range(10):
        assert monitor.update(_states(elbow_flex=0, wrist_flex=0)) == ""
    assert monitor.update(_states(elbow_flex=1, wrist_flex=1)) == ""
    # A brief disabled reading is ignored; a sustained one is a drop-out.
    assert monitor.update(_states(elbow_flex=0, wrist_flex=1)) == ""
    assert monitor.update(_states(elbow_flex=1, wrist_flex=1)) == ""
    assert monitor.update(_states(elbow_flex=0, wrist_flex=1)) == ""
    assert monitor.update(_states(elbow_flex=0, wrist_flex=1)) == ""
    assert "elbow_flex went from enabled to disabled" in monitor.update(_states(elbow_flex=0, wrist_flex=1))
    # A fault code is reported at once, by name.
    assert "wrist_flex reports under-voltage" in monitor.update(_states(elbow_flex=1, wrist_flex=0x9))
    assert monitor.update({"elbow_flex": None}) == ""


def test_motor_health_check_is_opt_in_and_stops_following(monkeypatch, caplog) -> None:
    """A motor that drops out mid-session: the loop keeps running, holds, and stops following."""
    from lerobot_teleoperator_rebot_vr.hardware import simulated
    from lerobot_teleoperator_rebot_vr.runtime import real

    assert _args().motor_health_check is False

    port = _free_port()
    args = _args(
        "--dry-run", "--motor-health-check", "--port", str(port), "--duration", "4",
        "--status-rate", "0.2", "--robot-port", "/not/opened", "--log-level", "WARNING",
        "--no-move-to-initial",
    )
    recorder = _Recorder()
    followers: list = []
    original = simulated.SimulatedFollower.__init__

    def keep(self, *a, **k):
        original(self, *a, **k)
        followers.append(self)

    monkeypatch.setattr(simulated.SimulatedFollower, "__init__", keep)
    errors: list = []

    def play() -> None:
        try:
            deadline = time.monotonic() + 20.0
            while True:
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            with connection:
                for packet in quest.handshake_packets():
                    connection.sendall(packet)
                started = time.monotonic()
                dropped = False
                while time.monotonic() - started < 3.0:
                    elapsed = time.monotonic() - started
                    # Grip released, then held while the hand keeps moving forward.
                    grip = 0.0 if elapsed < 0.3 else 1.0
                    state = replace(quest.REST_RIGHT, grip=grip, position=(0.2, 1.1, 0.4 + 0.05 * elapsed))
                    if elapsed > 1.5 and not dropped:
                        followers[0].motors["elbow_flex"].disable()  # supply dip: motor restarts disabled
                        dropped = True
                    tracking = quest.tracking_object(quest.REST_LEFT, state, time_stamp_ns=time.time_ns())
                    connection.sendall(quest.tracking_packet(tracking))
                    time.sleep(1.0 / 90.0)
        except Exception as exc:
            errors.append(exc)

    client = threading.Thread(target=play, daemon=True)
    client.start()
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        real._run_arm(args, observer=recorder)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    client.join(timeout=5.0)

    assert not errors, errors
    states = [state for _obs, _sent, state in recorder.steps]
    assert "active" in states
    last_active = len(states) - 1 - states[::-1].index("active")
    # Following ended while Grip was still held and frames were still arriving.
    assert set(states[last_active + 1:]) and "active" not in states[last_active + 1:]
    assert len(states) - last_active > 90
    assert "MOTOR PROBLEM: elbow_flex went from enabled to disabled" in caplog.text
