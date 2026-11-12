"""Print the selected controller's stream without connecting the robot.

Use it to check a headset before any motor is powered: that it connects, that
Tracking frames arrive and parse, which way the axes point, and which buttons
are which. Works with the PICO sender and with the XRoboToolkit Quest client;
both speak the same protocol and send the same right-handed frame.
"""

from __future__ import annotations

import argparse
import signal
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..config_rebot_vr import RebotVRConfig
from ..vr.controller import make_vr_controller


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=63901)
    parser.add_argument("--rate", type=float, default=10.0, help="print rate in Hz")
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until Ctrl-C")
    parser.add_argument("--grip-press", type=float, default=0.60,
                        help="Grip level from which the printed offset is measured, as in teleoperation")
    return parser


def _format_pose(position: object, quaternion: object) -> str:
    if position is None or quaternion is None:
        return "not provided by this sample"
    pos = np.asarray(position, dtype=float)
    quat = np.asarray(quaternion, dtype=float)
    rpy = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
    return (
        f"pos_m={np.array2string(pos, precision=4, separator=',')} "
        f"quat_xyzw={np.array2string(quat, precision=4, separator=',')} "
        f"rpy_deg={np.array2string(rpy, precision=2, separator=',')}"
    )


class GripOffset:
    """Motion since Grip was pressed, in the robot's frame: what the arm would be asked to do."""

    def __init__(self, press: float, release: float | None = None) -> None:
        self.press = float(press)
        self.release = self.press - 0.2 if release is None else float(release)
        self._reference: tuple[np.ndarray, Rotation] | None = None

    def update(self, action: dict) -> str:
        if not action["is_tracking"] or float(action["squeeze"]) <= self.release:
            self._reference = None
        position = np.asarray(action["grip_pos"], dtype=float)
        rotation = Rotation.from_quat(np.asarray(action["grip_quat"], dtype=float))
        if self._reference is None:
            if not action["is_tracking"] or float(action["squeeze"]) < self.press:
                return "  since Grip: Grip not held"
            self._reference = (position, rotation)
        offset = position - self._reference[0]
        turn = np.rad2deg((rotation * self._reference[1].inv()).as_rotvec())
        return (
            "  since Grip: "
            f"forward={offset[0]:+.3f} m  left={offset[1]:+.3f} m  up={offset[2]:+.3f} m  |  "
            f"turn about forward/left/up = {turn[0]:+.1f}/{turn[1]:+.1f}/{turn[2]:+.1f} deg"
        )


def _raw_extras(tracking: dict | None, hand: str) -> str:
    """Fields the arm does not use but that identify the sender and its buttons."""
    if not tracking:
        return ""
    controllers = tracking.get("Controller")
    controller = controllers.get(hand) if isinstance(controllers, dict) else None
    parts = []
    if isinstance(controller, dict):
        for label, key in (("menu", "menuButton"), ("stick_click", "axisClick")):
            if key in controller:
                parts.append(f"{label}={controller[key]}")
        if "axisX" in controller and "axisY" in controller:
            try:
                parts.append(f"stick=({float(controller['axisX']):+.2f},{float(controller['axisY']):+.2f})")
            except (TypeError, ValueError):
                pass
    if "Input" in tracking:
        parts.append(f"Input={tracking['Input']}")
    return "  ".join(parts)


def _stream_line(stats, previous, elapsed_s: float) -> str:
    frames_hz = (stats.tracking_frame_count - previous.tracking_frame_count) / max(elapsed_s, 1e-6)
    line = (
        f"link={'connected' if stats.connected else 'waiting for headset'} "
        f"frames={stats.tracking_frame_count} ({frames_hz:.0f}/s) "
        f"invalid={stats.invalid_frame_count} decode_warnings={stats.decode_warning_count}"
    )
    if stats.client_version:
        line += f" client={stats.client_version!r}"
    if stats.last_error and (
        stats.invalid_frame_count != previous.invalid_frame_count
        or stats.decode_warning_count != previous.decode_warning_count
    ):
        line += f"\n  ! {stats.last_error}"
    if stats.connected and stats.tracking_frame_count == previous.tracking_frame_count:
        if stats.invalid_frame_count != previous.invalid_frame_count:
            line += "\n  ! frames arrive but none parses; see the error above"
        else:
            line += "\n  ! connected but no Tracking frames: turn on Controller and Send in the headset app"
    return line


def main() -> None:
    args = _parser().parse_args()
    if args.rate <= 0.0 or args.duration < 0.0:
        raise ValueError("rate must be positive and duration must be non-negative")
    if not 0.2 < args.grip_press <= 1.0:
        raise ValueError("grip-press must be in (0.2, 1]")
    config = RebotVRConfig(
        hand_side=args.hand,
        ws_host=args.host,
        ws_port=args.port,
    )
    latest_tracking: dict = {}

    def on_tracking(_sample, tracking) -> None:
        latest_tracking["value"] = tracking

    controller = make_vr_controller(config, on_tracking=on_tracking)
    stop = False

    def stop_now(_signal_number, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, stop_now)
    signal.signal(signal.SIGTERM, stop_now)

    print(
        f"Listening for XRoboToolkit V1 on {args.host}:{args.port}; "
        "start the headset's Tracking sender (PICO app, or the XRoboToolkit Quest client).\n"
        "The robot is not connected. Poses below are in the robot's frame: X forward, Y left, Z up."
    )

    controller.connect()
    started = time.monotonic()
    offset = GripOffset(args.grip_press)
    previous_stats = controller.stats()
    previous_time = started
    try:
        while not stop:
            if args.duration and time.monotonic() - started >= args.duration:
                break
            action = controller.get_action()
            now = time.monotonic()
            stats = controller.stats()
            print(_stream_line(stats, previous_stats, now - previous_time))
            previous_stats, previous_time = stats, now
            print(
                f"tracking={bool(action['is_tracking'])} "
                f"grip={float(action['squeeze']):.3f} "
                f"trigger={float(action['trigger']):.3f} "
                f"primary={bool(action.get('primary_button', False))} "
                f"secondary={bool(action.get('secondary_button', False))}"
                + (
                    "  " + _raw_extras(latest_tracking.get("value"), args.hand)
                    if action["is_tracking"]
                    else ""
                )
            )
            print(f"  controller: {_format_pose(action['grip_pos'], action['grip_quat'])}")
            print(offset.update(action), flush=True)
            time.sleep(1.0 / args.rate)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
