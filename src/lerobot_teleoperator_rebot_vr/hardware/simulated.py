"""A stand-in follower for ``--dry-run``: no port, no CAN, no motors.

It offers the part of LeRobot's ``RebotB601Follower`` that the runner, the MIT
dispatcher and the shutdown sequence use, so the whole control path can run
with nothing attached. Each joint moves towards its last commanded position
with a first-order lag and a speed bound. That is enough to close the loop
(feedback follows commands) and says nothing about how the real arm tracks,
vibrates or loads its supply.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

STATUS_DISABLED, STATUS_ENABLED = 0, 1


class SimulatedMotor:
    """One joint, in radians, with the motor-handle calls the package makes."""

    def __init__(
        self,
        name: str,
        send_id: int,
        receive_id: int,
        *,
        position_rad: float = 0.0,
        time_constant_s: float = 0.08,
        max_speed_rad_s: float = 6.0,
    ) -> None:
        self.name = name
        self.send_id = int(send_id)
        self.receive_id = int(receive_id)
        self.position_rad = float(position_rad)
        self.velocity_rad_s = 0.0
        self.target_rad = float(position_rad)
        self.time_constant_s = float(time_constant_s)
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.enabled = False
        self.closed = False
        self.last_command: tuple[str, tuple[float, ...]] | None = None
        self.command_count = 0

    def _command(self, kind: str, position_rad: float, *rest: float) -> None:
        values = (float(position_rad), *map(float, rest))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{self.name}: non-finite {kind} command {values}")
        self.last_command = (kind, values)
        self.command_count += 1
        self.target_rad = float(position_rad)

    def send_mit(self, position_rad, velocity_rad_s, kp, kd, torque_nm) -> None:
        self._command("mit", position_rad, velocity_rad_s, kp, kd, torque_nm)

    def send_pos_vel(self, position_rad, velocity_rad_s) -> None:
        self._command("pos_vel", position_rad, velocity_rad_s)

    def send_force_pos(self, position_rad, velocity_rad_s, torque_ratio) -> None:
        self._command("force_pos", position_rad, velocity_rad_s, torque_ratio)

    def step(self, dt_s: float) -> None:
        if not self.enabled or dt_s <= 0.0:
            self.velocity_rad_s = 0.0
            return
        blend = 1.0 - math.exp(-dt_s / self.time_constant_s)
        bound = self.max_speed_rad_s * dt_s
        change = max(-bound, min(bound, (self.target_rad - self.position_rad) * blend))
        self.position_rad += change
        self.velocity_rad_s = change / dt_s

    def request_feedback(self) -> None:
        return None

    def get_state(self):
        return SimpleNamespace(
            pos=self.position_rad,
            vel=self.velocity_rad_s,
            torq=0.0,
            status_code=STATUS_ENABLED if self.enabled else STATUS_DISABLED,
            t_mos=25.0,
            t_rotor=25.0,
            can_id=self.send_id,
            arbitration_id=self.receive_id,
        )

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def ensure_mode(self, _mode) -> None:
        return None

    def clear_error(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _SimulatedBus:
    def __init__(self, follower: "SimulatedFollower") -> None:
        self._follower = follower
        self.closed = False

    def poll_feedback_once(self) -> None:
        self._follower.advance()

    def enable_all(self) -> None:
        for motor in self._follower.motors.values():
            motor.enable()

    def disable_all(self) -> None:
        for motor in self._follower.motors.values():
            motor.disable()

    def close_bus(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class SimulatedFollower:
    """Drop-in for ``RebotB601Follower`` that never touches hardware.

    ``config`` is the real follower's config object, so limits, CAN ids, modes
    and the relative-target bound are the ones a real run would use. Positions
    are degrees at this interface, as on the real follower.
    """

    name = "simulated_rebot_b601_follower"
    status_reports_torque_state = True

    def __init__(self, config: Any, *, start_deg: dict[str, float] | None = None) -> None:
        self.config = config
        self.motor_names = list(config.motor_can_ids.keys())
        self.cameras: dict[str, Any] = {}
        self.bus: _SimulatedBus | None = None
        self.motors: dict[str, SimulatedMotor] = {}
        self.is_calibrated = True
        self._start_deg = dict(start_deg or {})
        self._last_step_s: float | None = None

    @property
    def is_connected(self) -> bool:
        return self.bus is not None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # there is nothing to calibrate, and no zero to set
        if self.bus is not None:
            raise RuntimeError("simulated follower is already connected")
        self.bus = _SimulatedBus(self)
        for name, (send_id, receive_id) in self.config.motor_can_ids.items():
            self.motors[name] = SimulatedMotor(
                name,
                send_id,
                receive_id,
                position_rad=math.radians(self._start_deg.get(name, 0.0)),
            )
        self.bus.enable_all()
        self._last_step_s = time.monotonic()

    def advance(self, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else float(now_s)
        previous = now if self._last_step_s is None else self._last_step_s
        self._last_step_s = now
        for motor in self.motors.values():
            motor.step(min(0.1, max(0.0, now - previous)))

    def _present_pos(self) -> dict[str, float]:
        self.advance()
        return {name: math.degrees(motor.position_rad) for name, motor in self.motors.items()}

    def get_observation(self) -> dict[str, float]:
        if self.bus is None:
            raise RuntimeError("simulated follower is not connected")
        return {f"{name}.pos": value for name, value in self._present_pos().items()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """The real follower's position path: limits, then the relative bound."""
        if self.bus is None:
            raise RuntimeError("simulated follower is not connected")
        goal = {key.removesuffix(".pos"): float(value) for key, value in action.items() if key.endswith(".pos")}
        present = self._present_pos()
        limit = self.config.max_relative_target
        for name in list(goal):
            if name not in self.motors:
                del goal[name]
                continue
            if name in self.config.joint_limits:
                low, high = self.config.joint_limits[name]
                goal[name] = max(low, min(high, goal[name]))
            if limit is not None:
                bound = float(limit[name] if isinstance(limit, dict) else limit)
                goal[name] = present[name] + max(-bound, min(bound, goal[name] - present[name]))
            self.motors[name].send_pos_vel(math.radians(goal[name]), 0.0)
        return {f"{name}.pos": value for name, value in goal.items()}

    def disconnect(self) -> None:
        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.close()
        if self.bus is not None:
            self.bus.close()
        self.bus = None
        self.motors = {}


__all__ = ["SimulatedFollower", "SimulatedMotor", "STATUS_DISABLED", "STATUS_ENABLED"]
