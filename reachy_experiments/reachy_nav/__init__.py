"""Reachy2 right-arm navigation experiments (PyBullet).

The counterpart of ``ur5_nav``: scenes, kinematics and a collision layer for
getting r_arm_tip to goals that require routing *around* geometry. No planner
here -- that drops in against the same Env/Goal interface.
"""
from .envs import ALL, make            # noqa: F401
from .motion import self_pairs, straight_line, worst_contact   # noqa: F401
from .robot import (ARM_JOINTS, BASE_Z, GROUPS, L_ARM, R_ARM,   # noqa: F401
                    TIP_LINK, ReachyArm)
from .scene import Box, Env, Goal      # noqa: F401
