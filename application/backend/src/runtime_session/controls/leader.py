"""Moving a leader arm to where the follower is, so a human can take over without a jump.

Leader drivers only read: ``send_action`` refuses on a leader. So this drives the arm's
SDK under the driver, switching it to position control for the move and back to
following the human's hand when released.

shortcut: Trossen WidowX AI only, through its driver's private SDK handle; move it into the
plugins as a leader positioning method once the robot protocol has one, and support SO101 there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from physicalai.robot import Robot


def _trossen(leader: Robot) -> Any | None:
    """The Trossen SDK driver behind ``leader``, or ``None`` for any other arm."""
    driver = getattr(leader, "_driver", None)
    return driver if type(driver).__module__.startswith("trossen_arm") else None


def move_leader(leader: Robot, positions: np.ndarray, *, duration: float) -> bool:
    """Start moving ``leader`` to ``positions`` (as the leader reads them) over ``duration``; don't wait.

    The SDK interpolates from where the arm is. Returns ``False``, leaving the arm alone, for a
    leader that cannot be moved; the human brings it over by hand then.
    """
    driver = _trossen(leader)
    if driver is None:
        return False
    import trossen_arm

    # The driver reads joints in degrees and the gripper in its own units; the SDK takes radians.
    target = [
        float(value) if name == "gripper" else float(np.deg2rad(value))
        for name, value in zip(leader.joint_names, positions, strict=True)
    ]
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(trossen_arm.VectorDouble(target), duration, False)
    return True


def release_leader(leader: Robot) -> None:
    """Let the human move ``leader`` again, wherever its move got to."""
    if (driver := _trossen(leader)) is None:
        return
    import trossen_arm

    driver.set_all_modes(trossen_arm.Mode.external_effort)
    driver.set_all_external_efforts(trossen_arm.VectorDouble([0.0] * len(leader.joint_names)), 0.0, False)
