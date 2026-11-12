"""Show the simulated arm of a ``--dry-run`` in a browser, moving as it is commanded.

The dry run's follower is numbers in a log. This draws the arm from a URDF
with meshes (Seeed's ``ReBot_Arm_DM.urdf``; the package's own URDFs carry no
meshes) at the simulated joint angles, in a MeshCat page, so the response to
the controller can be watched. It shows where the simulated joints are, which
is where the commands sent them after a simple lag: not how the real arm
would track, vibrate or sag.

Needs the ``view`` extra (``pip install -e '.[view]'``).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

ARM_JOINTS = {
    "shoulder_pan": "joint1",
    "shoulder_lift": "joint2",
    "elbow_flex": "joint3",
    "wrist_flex": "joint4",
    "wrist_yaw": "joint5",
    "wrist_roll": "joint6",
}
# The gripper motor turns 0 (closed) to -270 degrees (open); each finger slides
# up to this far, in opposite directions.
GRIPPER_OPEN_RAD = math.radians(270.0)
FINGER_TRAVEL_M = 0.0285
FINGERS = {"finger_left": 1.0, "finger_right": -1.0}


def joint_configuration(model, motor_positions_rad: Mapping[str, float]):
    """The URDF model's configuration for these motor angles (radians)."""
    import numpy as np
    import pinocchio as pin

    q = pin.neutral(model)

    def put(joint_name: str, value: float) -> None:
        if model.existJointName(joint_name):
            joint = model.joints[model.getJointId(joint_name)]
            low, high = model.lowerPositionLimit[joint.idx_q], model.upperPositionLimit[joint.idx_q]
            q[joint.idx_q] = float(np.clip(value, low, high)) if low < high else value

    for motor, joint_name in ARM_JOINTS.items():
        if motor in motor_positions_rad:
            put(joint_name, motor_positions_rad[motor])
    if "gripper" in motor_positions_rad:
        opening = min(1.0, abs(motor_positions_rad["gripper"]) / GRIPPER_OPEN_RAD)
        for finger, direction in FINGERS.items():
            put(finger, direction * opening * FINGER_TRAVEL_M)
    return q


class ArmViewer:
    """A MeshCat page that follows ``read_positions()`` (motor name -> radians)."""

    def __init__(
        self,
        urdf: Path,
        read_positions: Callable[[], Mapping[str, float]],
        *,
        rate_hz: float = 30.0,
        open_browser: bool = True,
    ) -> None:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer

        urdf = Path(urdf).resolve()
        if not urdf.is_file():
            raise FileNotFoundError(f"viewer URDF not found: {urdf}")
        # Mesh paths in the URDF are relative to its own folder.
        model, collision, visual = pin.buildModelsFromUrdf(
            str(urdf), package_dirs=[str(urdf.parent), str(urdf.parent.parent)]
        )
        self.model = model
        self._read_positions = read_positions
        self._period_s = 1.0 / float(rate_hz)
        self._visualizer = MeshcatVisualizer(model, collision, visual)
        self._visualizer.initViewer(open=open_browser)
        self._visualizer.loadViewerModel()
        self._visualizer.display(pin.neutral(model))
        self.url = self._visualizer.viewer.url()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="arm-viewer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._period_s):
            try:
                self._visualizer.display(joint_configuration(self.model, self._read_positions()))
            except Exception as error:  # the viewer must never take the control loop down
                print(f"arm viewer stopped: {error}", flush=True)
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


__all__ = ["ARM_JOINTS", "ArmViewer", "joint_configuration"]
