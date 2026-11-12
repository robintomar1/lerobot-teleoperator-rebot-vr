"""The on-screen arm of a dry run: motor angles to the drawn model's joints."""

import math

import pytest

pin = pytest.importorskip("pinocchio")

from lerobot_teleoperator_rebot_vr.tools.arm_viewer import (  # noqa: E402
    FINGER_TRAVEL_M,
    joint_configuration,
)


def _model():
    """The drawn URDF's joints: six revolute arm joints and two sliding fingers."""
    model = pin.Model()
    parent = 0
    for index in range(1, 7):
        parent = model.addJoint(parent, pin.JointModelRZ(), pin.SE3.Identity(), f"joint{index}")
    for finger in ("finger_left", "finger_right"):
        model.addJoint(parent, pin.JointModelPX(), pin.SE3.Identity(), finger)
    return model


def _value(model, q, joint_name):
    return q[model.joints[model.getJointId(joint_name)].idx_q]


def test_arm_motors_set_their_joints_in_radians():
    model = _model()
    q = joint_configuration(model, {"shoulder_lift": -1.2, "wrist_roll": 0.4})
    assert _value(model, q, "joint2") == pytest.approx(-1.2)
    assert _value(model, q, "joint6") == pytest.approx(0.4)
    assert _value(model, q, "joint1") == 0.0


def test_gripper_motor_slides_the_fingers_apart():
    model = _model()
    closed = joint_configuration(model, {"gripper": 0.0})
    half = joint_configuration(model, {"gripper": math.radians(-135.0)})
    beyond = joint_configuration(model, {"gripper": math.radians(-400.0)})
    assert _value(model, closed, "finger_left") == 0.0
    assert _value(model, half, "finger_left") == pytest.approx(FINGER_TRAVEL_M / 2)
    assert _value(model, half, "finger_right") == pytest.approx(-FINGER_TRAVEL_M / 2)
    assert _value(model, beyond, "finger_left") == pytest.approx(FINGER_TRAVEL_M)


def test_view_urdf_needs_dry_run():
    """The viewer draws the simulated follower: asking for it on a real run is refused."""
    from lerobot_teleoperator_rebot_vr.runtime.cli import build_parser, validate_args

    args = build_parser().parse_args(["--view-urdf", "arm.urdf"])
    with pytest.raises(ValueError, match="needs --dry-run"):
        validate_args(args)
