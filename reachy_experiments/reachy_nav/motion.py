"""Collision evaluation and the straight-line baseline.

The headline metric of the UR5 harness is the gap between two numbers: how many
goals are *reachable* (a collision-free configuration exists) and how many are
reached by a naive straight line in joint space. The second number is the
planner-shaped hole. This module computes both.

Self-collision is handled the way ``Reachy2CBFPlanningDemo`` handles it: a
sphere pair is only checked if it has real headroom at the reference pose. A
fitted sphere model does not separate every pair the SRDF leaves enabled, and a
pair already touching at the start can never be made positive, so checking it
would report a permanent, meaningless collision.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np

from .robot import ReachyArm
from .scene import Env

DEFAULT_SRDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/reachy2.srdf")

SELF_HEADROOM = 0.020      # 20 mm, as the C++ demo's calibration uses
SELF_MARGIN = 0.005


@dataclass
class Contact:
    kind: str                  # "world" or "self"
    clearance: float
    detail: str


def srdf_disabled(srdf: Optional[Path] = None) -> Set[frozenset]:
    """Link pairs the SRDF declares non-colliding.

    The C++ demo builds its pair table the same way. Hand-rolling the rule
    instead is not enough: excluding only same-link pairs leaves adjacent links
    across a joint (``r_elbow_arm_link`` vs ``r_elbow_forearm_link``) and the
    mimic-coupled gripper (``r_hand_palm_link`` vs ``r_hand_proximal_link``) in
    the table, and both report permanent, meaningless collisions -- the second
    as a constant floor, because those fingers never move.
    """
    path = Path(srdf) if srdf is not None else DEFAULT_SRDF
    if not path.exists():
        return set()
    out = set()
    for e in ET.parse(str(path)).getroot().iter("disable_collisions"):
        out.add(frozenset((e.get("link1"), e.get("link2"))))
    return out


def movable_spheres(arm: ReachyArm, probe: float = 0.25) -> np.ndarray:
    """Indices of spheres the arm joints can actually move.

    Everything else -- wheels, lidar, the support bars, the other arm, the head
    -- is rigid for this planning group, so its world clearance is a constant
    and its self-pairs can never be influenced. Including them does not just
    waste work: a rigid pair sets a floor under the reported minimum, which is
    why an obstacle-free scene was reporting an identical +0.0155 m at every
    goal.
    """
    arm.zero()
    base = arm.sphere_centers()
    moved = np.zeros(len(arm.spheres), dtype=bool)
    for a in range(len(arm.arm)):
        q = np.zeros(len(arm.arm))
        q[a] = np.clip(probe, arm.lo[a], arm.hi[a]) or -probe
        arm.zero(); arm.set_config(q)
        moved |= np.linalg.norm(arm.sphere_centers() - base, axis=1) > 1e-6
    arm.zero()
    return np.flatnonzero(moved)


def self_pairs(arm: ReachyArm, reference: np.ndarray,
               srdf: Optional[Path] = None) -> np.ndarray:
    """Sphere pairs worth checking, calibrated at ``reference``.

    Returns an (P, 2) int array of indices into ``arm.spheres``.
    """
    arm.zero()
    arm.set_config(reference)
    c = arm.sphere_centers()
    r = arm.radii
    n = len(r)
    links = np.array([s.link for s in arm.spheres])
    mob = set(movable_spheres(arm).tolist())
    disabled = srdf_disabled(srdf)
    arm.zero(); arm.set_config(reference)
    c = arm.sphere_centers()
    keep = []
    for i in range(n):
        for j in range(i + 1, n):
            if links[i] == links[j]:
                continue                      # same link: rigid, never changes
            if i not in mob and j not in mob:
                continue                      # neither endpoint moves: constant
            if frozenset((links[i], links[j])) in disabled:
                continue                      # SRDF says this pair never collides
            gap = np.linalg.norm(c[i] - c[j]) - r[i] - r[j]
            if gap >= SELF_HEADROOM:
                keep.append((i, j))
    return np.asarray(keep, dtype=int)


def worst_contact(arm: ReachyArm, env: Env, q: np.ndarray,
                  pairs: Optional[np.ndarray] = None,
                  mobile: Optional[np.ndarray] = None) -> Contact:
    """Smallest clearance at ``q`` over world and (optionally) self channels.

    ``mobile`` restricts the world channel to spheres the arm can move; the
    rest are rigid and their clearance is a property of the scene, checked once
    when it is built rather than at every configuration.

    Deliberately does **not** call ``arm.zero()``: only the group's joints are
    ever written, every other joint stays at the zero it was loaded with, and
    resetting all 34 of them here cost more than the whole rest of the check
    (0.78 ms of 0.85 ms on dual_arm). This is the planner's inner loop.
    """
    arm.set_config(q)
    c = arm.sphere_centers()
    r = arm.radii
    sel = mobile if mobile is not None else np.arange(len(r))
    world = env.clearance(c[sel], r[sel])
    k = int(np.argmin(world))
    worst = Contact("world", float(world[k]), arm.spheres[sel[k]].link)
    if pairs is not None and len(pairs):
        d = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)
        gap = d - r[pairs[:, 0]] - r[pairs[:, 1]] - SELF_MARGIN
        m = int(np.argmin(gap))
        if gap[m] < worst.clearance:
            worst = Contact("self", float(gap[m]),
                            f"{arm.spheres[pairs[m,0]].link} vs "
                            f"{arm.spheres[pairs[m,1]].link}")
    return worst


def straight_line(arm: ReachyArm, env: Env, start: np.ndarray, goal: np.ndarray,
                  pairs: Optional[np.ndarray] = None,
                  mobile: Optional[np.ndarray] = None,
                  resolution: float = 0.024) -> Tuple[bool, float, int, int]:
    """Densely check the straight joint-space segment ``start`` -> ``goal``.

    ``resolution`` is the 0.024 rad the C++ demos audit at, so a collision
    reported here is one the planner rows would also see.

    Returns ``(collides, min_clearance, first_bad_step, n_steps)``.
    """
    span = float(np.max(np.abs(goal - start)))
    n = max(2, int(np.ceil(span / resolution)) + 1)
    lo, first = np.inf, -1
    for k in range(n):
        q = start + (goal - start) * (k / (n - 1))
        c = worst_contact(arm, env, q, pairs, mobile)
        if c.clearance < lo:
            lo = c.clearance
        if c.clearance < 0.0 and first < 0:
            first = k
    return (first >= 0), lo, first, n
