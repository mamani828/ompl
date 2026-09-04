"""Six obstacle scenes for Reachy2's right arm.

These are the ``ur5_nav`` scenes re-sited, not rescaled. Three measurements
force that, all taken from the r_arm_tip envelope over 30000 uniform in-limits
samples (see ``scripts/reach_envelope.py``):

    x: -0.661 .. +0.646   p5 -0.453  p50 +0.046  p95 +0.492
    y: -0.858 .. +0.247   p5 -0.808  p50 -0.479  p95 -0.005
    z: +0.461 .. +1.763   p5 +0.688  p50 +1.195  p95 +1.617

- **Forward reach falls off a cliff.** x > 0.45 is 7.3% of samples, x > 0.60 is
  0.9%, and x > 0.65 does not occur. UR5's scenes sit at x up to 0.7 on a
  0.9144 m table; placed there for Reachy they are simply scenery.
- **The workspace is one-sided.** The right arm is centred at y = -0.48 and
  crosses the midline only to y = +0.247, so the UR5's y-symmetric scenes waste
  half their geometry. Every scene here is biased to -y.
- **There is no table.** Reachy stands on the floor with base_link at z=0.1075,
  so obstacles are floor-standing and goal heights come from the arm's own
  z band rather than from a table top.

Goal heights cluster near z = 1.0-1.4, which is the p25-p75 of the envelope and
also roughly Reachy's chest-to-eye height -- the band a bimanual robot is built
to work in.
"""
from __future__ import annotations

import numpy as np

from .scene import Box, Env, Goal

WOOD = (0.55, 0.35, 0.18, 1.0)
GREY = (0.40, 0.42, 0.45, 1.0)

# Centre of the right arm's comfortable band, from the envelope above.
Y_MID = -0.30
Z_MID = 1.15

# r_arm_tip at the all-zero home configuration, measured, not assumed. The
# clutter scene places obstacles along the segment from here to each goal.
HOME_TIP = np.array([0.0197, -0.3682, 0.4660])


def env_empty() -> Env:
    """Baseline: nothing in the way. Checks reach and IK only."""
    return Env(
        name="empty",
        description="No obstacles; baseline reach and IK check.",
        boxes=[],
        goals=[
            Goal([0.42, -0.20, 1.15], "+x", "front, near midline"),
            Goal([0.40, -0.55, 1.10], "+x", "front, outboard"),
            Goal([0.25, -0.62, 1.40], "-z", "high outboard, top-down"),
            Goal([0.10, -0.45, 0.85], "-z", "low, close in"),
        ],
    )


def env_shelf() -> Env:
    """Two-bay shelf. Goals sit inside slots, so the wrist enters from +x.

    Kept at the C++ demo's shelf x = 0.62 and 0.14 m depth so the two agree,
    but the bays are re-centred on the right arm's y band instead of y = 0.
    """
    # x = 0.62 is the C++ demo's shelf, but that is the very edge of this
    # arm's reach: the outboard bay goals there failed IK by 11-26 mm. 0.53
    # keeps the same slot geometry inside the envelope (x > 0.55 is 2.6% of
    # samples, x > 0.45 is 7.3%).
    shelf_x, depth, width = 0.53, 0.12, 0.46
    y_c = -0.32
    bottom, pitch, top = 0.80, 0.30, 1.70
    boxes = [
        Box([shelf_x + depth, y_c, top / 2], [0.02, width / 2, top / 2], WOOD),
        Box([shelf_x, y_c + width / 2, top / 2], [depth, 0.02, top / 2], WOOD),
        Box([shelf_x, y_c - width / 2, top / 2], [depth, 0.02, top / 2], WOOD),
    ]
    for k in range(3):
        boxes.append(Box([shelf_x, y_c, bottom + k * pitch],
                         [depth, width / 2, 0.015], WOOD))
    return Env(
        name="shelf",
        description="Two-bay shelf; goals require entering horizontal slots.",
        boxes=boxes,
        goals=[
            Goal([shelf_x, y_c - 0.10, bottom + pitch / 2], "+x", "lower bay, outboard"),
            Goal([shelf_x, y_c + 0.10, bottom + pitch / 2], "+x", "lower bay, inboard"),
            Goal([shelf_x, y_c - 0.08, bottom + 1.5 * pitch], "+x", "upper bay, outboard"),
            Goal([shelf_x, y_c + 0.08, bottom + 1.5 * pitch], "+x", "upper bay, inboard"),
        ],
    )


def env_wall_gap() -> Env:
    """A wall across the arm's reach with a single window through it."""
    wall_x = 0.40
    gap_y, gap_z = Y_MID, 1.15
    gap_w, gap_h = 0.30, 0.28
    y_lo, y_hi = -0.90, 0.25
    z_top = 1.75
    boxes = [
        Box([wall_x, (y_lo + gap_y - gap_w / 2) / 2, z_top / 2],
            [0.02, (gap_y - gap_w / 2 - y_lo) / 2, z_top / 2], GREY),
        Box([wall_x, (y_hi + gap_y + gap_w / 2) / 2, z_top / 2],
            [0.02, (y_hi - gap_y - gap_w / 2) / 2, z_top / 2], GREY),
        Box([wall_x, gap_y, (gap_z - gap_h / 2) / 2],
            [0.02, gap_w / 2, (gap_z - gap_h / 2) / 2], GREY),
        Box([wall_x, gap_y, (gap_z + gap_h / 2 + z_top) / 2],
            [0.02, gap_w / 2, (z_top - gap_z - gap_h / 2) / 2], GREY),
    ]
    return Env(
        name="wall_gap",
        description="Full wall with one window; the window is the only route.",
        boxes=boxes,
        goals=[
            Goal([0.55, gap_y, gap_z], "+x", "through the window"),
            Goal([0.52, gap_y - 0.08, gap_z + 0.06], "+x", "through, offset"),
        ],
    )


def env_corridor() -> Env:
    """Two walls forming a slot the arm must traverse along +x."""
    half_gap = 0.13
    y_c = -0.32
    z_top = 1.60
    boxes = [
        Box([0.42, y_c - half_gap - 0.02, z_top / 2], [0.22, 0.02, z_top / 2], GREY),
        Box([0.42, y_c + half_gap + 0.02, z_top / 2], [0.22, 0.02, z_top / 2], GREY),
        Box([0.52, y_c, 1.45], [0.12, half_gap, 0.02], GREY),
    ]
    return Env(
        name="corridor",
        description="Slot between two walls, roofed over the far third.",
        boxes=boxes,
        goals=[
            Goal([0.55, y_c, 1.20], "+x", "far end of the slot"),
            Goal([0.55, y_c, 1.00], "+x", "far end, low"),
        ],
    )


def env_pillars(seed: int = 0) -> Env:
    """Randomised posts with top-down goals between them.

    Posts are rejected if they come within ``clear`` of a goal or of the
    vertical approach above it, so every goal stays reachable for any seed --
    the same guarantee ur5_nav makes, for the same reason: a post landing on a
    goal makes the scene unsolvable rather than hard.
    """
    rng = np.random.default_rng(seed)
    goals = [
        Goal([0.36, -0.20, 1.05], "-z", "between posts, inboard"),
        Goal([0.30, -0.52, 1.05], "-z", "between posts, outboard"),
        Goal([0.45, -0.36, 1.05], "-z", "between posts, far"),
    ]
    # Keep every post clear of each goal *and* of the vertical column above it,
    # which is the approach corridor for a "-z" goal. A post that clips that
    # column makes the scene unsolvable rather than hard. The earlier test only
    # compared centres and let a post land 1.3 mm inside a goal configuration.
    clear = 0.17
    post_half = 0.028
    boxes, tries = [], 0
    while len(boxes) < 7 and tries < 8000:
        tries += 1
        x = rng.uniform(0.16, 0.50)
        y = rng.uniform(-0.70, -0.04)
        h = rng.uniform(0.30, 0.58)
        cz = 0.80 + h / 2
        ok = True
        for g in goals:
            gx, gy, _ = g.position
            if np.hypot(gx - x, gy - y) < clear + post_half:
                ok = False
                break
        if ok:
            boxes.append(Box([x, y, cz], [post_half, post_half, h / 2], GREY))
    return Env(
        name="pillars",
        description=f"7 randomised posts (seed {seed}); top-down goals between them.",
        boxes=boxes,
        goals=goals,
    )


def env_clutter(seed: int = 0) -> Env:
    """Randomised boxes at mixed heights, including overhead obstacles."""
    rng = np.random.default_rng(1000 + seed)
    goals = [
        Goal([0.40, -0.24, 1.12], "+x", "behind clutter, inboard"),
        Goal([0.34, -0.56, 1.05], "+x", "behind clutter, outboard"),
        Goal([0.28, -0.38, 1.42], "-z", "above clutter"),
    ]
    # Purely random boxes left every goal reachable in a straight line (0/3),
    # so the scene measured nothing. Seed it with obstacles placed *on* the
    # workspace segment from the home tip to each goal -- the thing a naive
    # joint-space interpolation roughly follows -- then fill in at random.
    clear = 0.16
    boxes = []
    for g in goals:
        gp = np.asarray(g.position, dtype=float)
        for frac in (0.55, 0.78):
            c = HOME_TIP + (gp - HOME_TIP) * frac
            c = c + rng.normal(0.0, 0.02, 3)
            half = rng.uniform(0.040, 0.070, 3)
            if np.linalg.norm(c - gp) > clear + half.max():
                boxes.append(Box(c.tolist(), half.tolist(), GREY))
    tries = 0
    while len(boxes) < 10 and tries < 6000:
        tries += 1
        c = np.array([rng.uniform(0.14, 0.50), rng.uniform(-0.72, 0.02),
                      rng.uniform(0.85, 1.55)])
        half = rng.uniform(0.030, 0.070, 3)
        if all(np.any(np.abs(c - np.asarray(g.position)) > clear + half.max())
               for g in goals):
            boxes.append(Box(c.tolist(), half.tolist(), GREY))
    return Env(
        name="clutter",
        description=f"10 randomised boxes at mixed heights (seed {seed}).",
        boxes=boxes,
        goals=goals,
    )


REGISTRY = {
    "empty": lambda seed=0: env_empty(),
    "shelf": lambda seed=0: env_shelf(),
    "wall_gap": lambda seed=0: env_wall_gap(),
    "corridor": lambda seed=0: env_corridor(),
    "pillars": env_pillars,
    "clutter": env_clutter,
}
ALL = list(REGISTRY)


def make(name: str, seed: int = 0) -> Env:
    if name not in REGISTRY:
        raise KeyError(f"unknown env {name!r}; have {ALL}")
    return REGISTRY[name](seed)
