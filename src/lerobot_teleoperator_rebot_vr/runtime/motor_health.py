"""Notice a Damiao motor that has stopped holding the arm.

The follower's feedback check only looks at joint angles. A motor that has
protected itself, or that restarted after a supply dip and came back disabled,
still reports a perfectly good angle, so nothing else in the loop sees it.

This reads only the state the SDK already caches; it sends nothing on the bus.
It cannot see a motor that has lost power altogether: with no replies the
cache keeps the last state, which still says "enabled".
"""

from __future__ import annotations

from collections.abc import Mapping

STATUS_DISABLED, STATUS_ENABLED = 0, 1

# Status codes a Damiao motor reports once it has protected itself. It produces
# no torque until the error is cleared.
DAMIAO_FAULTS = {
    0x8: "over-voltage",
    0x9: "under-voltage",
    0xA: "over-current",
    0xB: "driver over-temperature",
    0xC: "winding over-temperature",
    0xD: "communication lost",
    0xE: "overload",
}


class MotorHealthMonitor:
    """Turn cached motor states into a description of what is wrong, or ""."""

    def __init__(self, dropout_reads: int = 5) -> None:
        if dropout_reads <= 0:
            raise ValueError("dropout_reads must be positive")
        # A motor reads as disabled for a moment after it is enabled; several
        # reads in a row, after it has been seen enabled, is not that.
        self.dropout_reads = int(dropout_reads)
        self._disabled_reads: dict[str, int] = {}

    def update(self, states: Mapping[str, object]) -> str:
        faults, dropped = [], []
        for name, state in states.items():
            code = getattr(state, "status_code", None)
            if code in DAMIAO_FAULTS:
                faults.append(f"{name} reports {DAMIAO_FAULTS[code]}")
            elif code == STATUS_ENABLED:
                self._disabled_reads[name] = 0
            elif code == STATUS_DISABLED and name in self._disabled_reads:
                self._disabled_reads[name] += 1
                if self._disabled_reads[name] >= self.dropout_reads:
                    dropped.append(name)
        problems = []
        if faults:
            problems.append("; ".join(faults))
        if dropped:
            problems.append(
                ", ".join(dropped)
                + " went from enabled to disabled (what a motor does when its supply dips and it restarts)"
            )
        return "; ".join(problems)


def read_motor_health(monitor: MotorHealthMonitor, robot) -> str:
    """Never let a diagnostic break the control loop."""
    try:
        return monitor.update({name: motor.get_state() for name, motor in robot.motors.items()})
    except Exception:
        return ""


__all__ = ["DAMIAO_FAULTS", "MotorHealthMonitor", "read_motor_health"]
