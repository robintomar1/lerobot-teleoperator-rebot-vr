"""Replay Quest-shaped XRoboToolkit packets at the V1 server, with no headset.

This reproduces, byte for byte where it matters, what
XR-Robotics/XRoboToolkit-Unity-Client-Quest v1.0.1 puts on the wire, so the PC
pipeline (parse -> clutch -> IK -> joint targets) can be exercised offline:

- ``Network/PackageHandle.cs`` ``Pack``: ``0x3F``, command, little-endian int32
  body length, body, little-endian int64 Unix milliseconds, ``0xA5``.
- ``Network/TcpHandler.cs``: on connect the client sends ``CONNECT`` (0x19,
  ``"<sn>|-1"``) and ``VERSION`` (0x6C, ``"<sn>|1.0|<app version>"``), then a
  ``HEARTBEAT`` (0x23) every 10 s. It needs no reply from the PC. Tracking goes
  out as ``FUNCTION`` (0x6D) with ``{"functionName": "Tracking", "value":
  "<json string>"}``.
- ``UI/UIOperate.cs``: the serial number is the constant ``"TestDevice"``.
- ``TrackingData.cs``: the keys of the Tracking object, and the pose. A pose is
  the Unity world pose of the controller, converted from Unity's left-handed
  frame by ``ConvertHandedness`` (``z``, ``qz`` and ``qw`` negated), written as
  ``"x,y,z,qx,qy,qz,qw"``.

Scenarios are written in Unity's frame (X right, Y up, Z forward, left-handed),
the frame the headset works in, and go through the same conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import struct
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation

SEND_PACKET_HEAD = 0x3F
PACKET_END = 0xA5
CMD_CONNECT = 0x19
CMD_VERSION = 0x6C
CMD_HEARTBEAT = 0x23
CMD_FUNCTION = 0x6D

QUEST_DEVICE_SN = "TestDevice"
QUEST_APP_VERSION = "1.0.1"
HEARTBEAT_INTERVAL_S = 10.0
# OVRPlugin.systemDisplayFrequency is set to 90 and one Tracking message is
# queued per rendered frame.
DEFAULT_RATE_HZ = 90.0


def pack_client_packet(command: int, body: bytes, *, timestamp_ms: int | None = None) -> bytes:
    """One headset-to-PC packet, as ``PackageHandle.Pack`` builds it."""
    if not 0 <= int(command) <= 0xFF:
        raise ValueError("command must fit in one byte")
    stamp = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    return b"".join(
        (
            struct.pack("<BBi", SEND_PACKET_HEAD, int(command), len(body)),
            body,
            struct.pack("<qB", stamp, PACKET_END),
        )
    )


def unity_to_wire_pose(position, quaternion_xyzw) -> tuple[np.ndarray, np.ndarray]:
    """``TrackingData.ConvertHandedness``: Unity left-handed to right-handed."""
    wire_position = np.asarray(position, dtype=np.float64).copy()
    wire_quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).copy()
    wire_position[2] *= -1.0
    wire_quaternion[2] *= -1.0
    wire_quaternion[3] *= -1.0
    return wire_position, wire_quaternion


def format_single(value: float) -> str:
    """A C# ``float`` printed with ``ToString("R")`` (invariant culture)."""
    single = np.float32(value)
    if single == 0.0:
        return "-0" if math.copysign(1.0, float(single)) < 0 else "0"
    if 1e-5 <= abs(float(single)) < 1e15:
        return np.format_float_positional(single, unique=True, trim="-")
    mantissa, exponent = np.format_float_scientific(single, unique=True, trim="-").split("e")
    return f"{mantissa}E{'-' if exponent.startswith('-') else '+'}{exponent.lstrip('+-').zfill(2)}"


def pose_string(position, quaternion_xyzw) -> str:
    """``TrackingData.GetPoseStr`` applied to a Unity-frame pose."""
    wire_position, wire_quaternion = unity_to_wire_pose(position, quaternion_xyzw)
    return ",".join(format_single(value) for value in (*wire_position, *wire_quaternion))


@dataclass(frozen=True)
class ControllerState:
    """One controller, in Unity's frame: X right, Y up, Z forward, metres."""

    position: tuple[float, float, float] = (0.2, 1.1, 0.4)
    quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    grip: float = 0.0
    trigger: float = 0.0
    primary_button: bool = False
    secondary_button: bool = False
    menu_button: bool = False
    axis_x: float = 0.0
    axis_y: float = 0.0
    axis_click: bool = False

    def to_json(self) -> dict[str, object]:
        # Key order of TrackingData.GetControllerJsonData, then the pose.
        return {
            "axisX": float(np.float32(self.axis_x)),
            "axisY": float(np.float32(self.axis_y)),
            "axisClick": bool(self.axis_click),
            "grip": float(np.float32(self.grip)),
            "trigger": float(np.float32(self.trigger)),
            "primaryButton": bool(self.primary_button),
            "secondaryButton": bool(self.secondary_button),
            "menuButton": bool(self.menu_button),
            "pose": pose_string(self.position, self.quaternion_xyzw),
        }


REST_LEFT = ControllerState(position=(-0.2, 1.1, 0.4))
REST_RIGHT = ControllerState(position=(0.2, 1.1, 0.4))


def tracking_object(
    left: ControllerState,
    right: ControllerState,
    *,
    time_stamp_ns: int,
    predict_time_us: float = 0.0,
    head: bool = False,
    input_device: int = 1,
) -> dict[str, object]:
    """The object ``TrackingData.Get`` fills, with the Controller toggle on."""
    data: dict[str, object] = {"predictTime": float(predict_time_us), "appState": {"focus": True}}
    if head:
        data["Head"] = {"pose": pose_string((0.0, 1.6, 0.0), (0.0, 0.0, 0.0, 1.0)), "status": 3}
    data["Controller"] = {"left": left.to_json(), "right": right.to_json()}
    data["timeStampNs"] = int(time_stamp_ns)
    data["Input"] = int(input_device)
    return data


def tracking_packet(tracking: dict[str, object], *, timestamp_ms: int | None = None) -> bytes:
    """A Tracking message: the object is JSON inside a JSON string, as sent."""
    separators = (",", ":")
    body = json.dumps(
        {"functionName": "Tracking", "value": json.dumps(tracking, separators=separators)},
        separators=separators,
    )
    return pack_client_packet(CMD_FUNCTION, body.encode("utf-8"), timestamp_ms=timestamp_ms)


def handshake_packets(
    serial_number: str = QUEST_DEVICE_SN, app_version: str = QUEST_APP_VERSION
) -> list[bytes]:
    """What ``TcpHandler.ConnectInit`` sends once the socket is up."""
    return [
        pack_client_packet(CMD_CONNECT, f"{serial_number}|-1".encode()),
        pack_client_packet(CMD_VERSION, f"{serial_number}|1.0|{app_version}".encode()),
    ]


def heartbeat_packet(serial_number: str = QUEST_DEVICE_SN) -> bytes:
    return pack_client_packet(CMD_HEARTBEAT, serial_number.encode())


# --- scenarios -------------------------------------------------------------


def _smooth(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return fraction * fraction * (3.0 - 2.0 * fraction)


@dataclass(frozen=True)
class Segment:
    """Hold or change the driven controller for ``duration_s``."""

    duration_s: float
    label: str
    update: Callable[[ControllerState, float], ControllerState] = field(
        default=lambda state, _fraction: state
    )


def _hold(duration_s: float, label: str) -> Segment:
    return Segment(duration_s, label)


def _set(label: str, duration_s: float = 0.3, **changes) -> Segment:
    return Segment(duration_s, label, lambda state, _fraction: replace(state, **changes))


def _translate(duration_s: float, label: str, offset) -> Segment:
    offset = np.asarray(offset, dtype=np.float64)
    start: list[np.ndarray] = []

    def update(state: ControllerState, fraction: float) -> ControllerState:
        if not start:
            start.append(np.asarray(state.position, dtype=np.float64))
        return replace(state, position=tuple(start[0] + _smooth(fraction) * offset))

    return Segment(duration_s, label, update)


def _rotate(duration_s: float, label: str, axis, angle_deg: float) -> Segment:
    """Rotate about a Unity world axis. Unity rotations are left-handed:
    a positive angle about +Y turns the controller to the right."""
    axis = np.asarray(axis, dtype=np.float64)
    start: list[np.ndarray] = []

    def update(state: ControllerState, fraction: float) -> ControllerState:
        if not start:
            start.append(np.asarray(state.quaternion_xyzw, dtype=np.float64))
        # The components of a left-handed rotation of `angle` about `axis` are
        # those of a quaternion with the same (axis, angle) numbers.
        delta = Rotation.from_rotvec(axis * math.radians(angle_deg) * _smooth(fraction))
        quaternion = (delta * Rotation.from_quat(start[0])).as_quat()
        return replace(state, quaternion_xyzw=tuple(quaternion))

    return Segment(duration_s, label, update)


def _ramp(duration_s: float, label: str, name: str, target: float) -> Segment:
    start: list[float] = []

    def update(state: ControllerState, fraction: float) -> ControllerState:
        if not start:
            start.append(float(getattr(state, name)))
        return replace(state, **{name: start[0] + _smooth(fraction) * (target - start[0])})

    return Segment(duration_s, label, update)


def axes_scenario(amplitude_m: float = 0.10, angle_deg: float = 30.0) -> list[Segment]:
    """Grip, then one move along each axis and one turn, then the buttons."""
    there, back = 1.5, 1.5
    return [
        _hold(1.0, "streaming, Grip released"),
        _ramp(0.3, "press Grip", "grip", 1.0),
        _hold(0.7, "Grip held, controller still"),
        _translate(there, "move FORWARD (Unity +Z)", (0, 0, amplitude_m)),
        _translate(back, "back", (0, 0, -amplitude_m)),
        _translate(there, "move RIGHT (Unity +X)", (amplitude_m, 0, 0)),
        _translate(back, "back", (-amplitude_m, 0, 0)),
        _translate(there, "move UP (Unity +Y)", (0, amplitude_m, 0)),
        _translate(back, "back", (0, -amplitude_m, 0)),
        _rotate(there, "turn RIGHT about vertical (Unity +Y)", (0, 1, 0), angle_deg),
        _rotate(back, "back", (0, 1, 0), -angle_deg),
        _ramp(0.5, "squeeze Trigger", "trigger", 1.0),
        _ramp(0.5, "release Trigger", "trigger", 0.0),
        _ramp(0.3, "release Grip", "grip", 0.0),
        _hold(0.7, "Grip released"),
        _set("press A/X (primary)", primary_button=True),
        _set("release A/X", primary_button=False),
        _hold(2.5, "return to initial pose runs"),
        _set("press B/Y (secondary)", secondary_button=True),
        _set("release B/Y", secondary_button=False),
        _hold(4.0, "return to zero runs"),
    ]


def jump_scenario(amplitude_m: float = 0.10, angle_deg: float = 30.0) -> list[Segment]:
    """Grip held while the reported pose teleports, as when tracking is lost."""
    del angle_deg
    return [
        _hold(1.0, "streaming, Grip released"),
        _ramp(0.3, "press Grip", "grip", 1.0),
        _hold(0.7, "Grip held, controller still"),
        _translate(1.0, "move FORWARD (Unity +Z)", (0, 0, amplitude_m)),
        _set("pose JUMPS to the origin, Grip still held", duration_s=2.0, position=(0.0, 0.0, 0.0)),
        _ramp(0.3, "release Grip", "grip", 0.0),
        _hold(1.0, "Grip released"),
    ]


def idle_scenario(amplitude_m: float = 0.10, angle_deg: float = 30.0) -> list[Segment]:
    del amplitude_m, angle_deg
    return [_hold(5.0, "streaming, Grip released")]


SCENARIOS: dict[str, Callable[..., list[Segment]]] = {
    "axes": axes_scenario,
    "jump": jump_scenario,
    "idle": idle_scenario,
}


def scenario_states(
    segments: list[Segment], rate_hz: float, initial: ControllerState
) -> Iterator[tuple[float, str, ControllerState]]:
    """(time, segment label, state) for every frame of a scenario."""
    state = initial
    elapsed = 0.0
    step = 1.0 / rate_hz
    for segment in segments:
        frames = max(1, round(segment.duration_s * rate_hz))
        for index in range(1, frames + 1):
            state = segment.update(state, index / frames)
            elapsed += step
            yield elapsed, segment.label, state


def scenario_packets(
    name: str = "axes",
    *,
    hand: str = "right",
    rate_hz: float = DEFAULT_RATE_HZ,
    amplitude_m: float = 0.10,
    angle_deg: float = 30.0,
    start_ns: int = 1_700_000_000_000_000_000,
) -> Iterator[tuple[float, str, bytes]]:
    """(time, label, Tracking packet) for a whole scenario, for offline tests."""
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    segments = SCENARIOS[name](amplitude_m, angle_deg)
    initial = REST_RIGHT if hand == "right" else REST_LEFT
    for elapsed, label, state in scenario_states(segments, rate_hz, initial):
        left, right = (REST_LEFT, state) if hand == "right" else (state, REST_RIGHT)
        tracking = tracking_object(
            left,
            right,
            time_stamp_ns=start_ns + int(elapsed * 1e9),
            predict_time_us=elapsed * 1e6,
        )
        yield elapsed, label, tracking_packet(tracking)


# --- TCP client ------------------------------------------------------------


def run_client(
    host: str,
    port: int,
    *,
    scenario: str,
    hand: str,
    rate_hz: float,
    amplitude_m: float,
    angle_deg: float,
    loop: bool,
    connect_timeout_s: float,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] = print,
) -> int:
    """Connect like the headset and play the scenario in real time. Returns frames sent."""
    deadline = time.monotonic() + connect_timeout_s
    while True:
        try:
            connection = socket.create_connection((host, port), timeout=2.0)
            break
        except OSError as exc:
            if time.monotonic() >= deadline or should_stop():
                raise ConnectionError(f"no V1 server at {host}:{port}: {exc}") from exc
            time.sleep(0.5)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    log(f"connected to {host}:{port} as {QUEST_DEVICE_SN} (synthetic Quest client)")
    sent = 0
    try:
        for packet in handshake_packets():
            connection.sendall(packet)
        last_heartbeat = time.monotonic()
        while not should_stop():
            started = time.monotonic()
            last_label = None
            for elapsed, label, packet in scenario_packets(
                scenario,
                hand=hand,
                rate_hz=rate_hz,
                amplitude_m=amplitude_m,
                angle_deg=angle_deg,
                start_ns=time.time_ns(),
            ):
                if should_stop():
                    break
                if label != last_label:
                    log(f"[{elapsed:6.2f}s] {label}")
                    last_label = label
                connection.sendall(packet)
                sent += 1
                if time.monotonic() - last_heartbeat > HEARTBEAT_INTERVAL_S:
                    connection.sendall(heartbeat_packet())
                    last_heartbeat = time.monotonic()
                delay = started + elapsed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            if not loop:
                break
    finally:
        connection.close()
    log(f"sent {sent} Tracking frames")
    return sent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="PC running the V1 server")
    parser.add_argument("--port", type=int, default=63901)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="axes")
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help="Tracking frames per second")
    parser.add_argument("--amplitude", type=float, default=0.10, help="translation per move, metres")
    parser.add_argument("--angle", type=float, default=30.0, help="rotation per turn, degrees")
    parser.add_argument("--loop", action="store_true", help="repeat until Ctrl-C")
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.rate <= 0 or args.amplitude < 0:
        raise ValueError("rate must be positive and amplitude non-negative")
    stop = False

    def stop_now(_signal_number, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_now)
    signal.signal(signal.SIGTERM, stop_now)
    run_client(
        args.host,
        args.port,
        scenario=args.scenario,
        hand=args.hand,
        rate_hz=args.rate,
        amplitude_m=args.amplitude,
        angle_deg=args.angle,
        loop=args.loop,
        connect_timeout_s=args.connect_timeout,
        should_stop=lambda: stop,
    )


if __name__ == "__main__":
    main()
