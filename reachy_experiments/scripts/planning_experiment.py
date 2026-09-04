#!/usr/bin/env python3
"""Plan against the cached GP-SDF; score every result against ground truth.

    python scripts/cache_sdf.py --out out/shelf_map.npz
    python scripts/planning_experiment.py --map out/shelf_map.npz --goals 20

Goals are sampled and then **kept only if they are collision-free in truth**,
so every one of them is a motion the robot could legitimately make. The planner
never sees truth. Two independent failure directions are then counted:

- **conservative** -- truth says the goal is free, the map refuses it. Nothing
  collides; routes simply cease to exist, and the caller sees an unreachable
  pose rather than a mapping problem.
- **dangerous** -- the map plans a path that penetrates real geometry, while
  reporting positive clearance the whole way.

A straight-line baseline is scored alongside, so the planner's contribution is
separable from the map's.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_nav import ReachyArm                                   # noqa: E402
from reachy_nav import planner as pl                               # noqa: E402
from reachy_nav.live_sdf import GridCache                          # noqa: E402
from reachy_nav.motion import (movable_spheres, self_pairs,        # noqa: E402
                               straight_line, worst_contact)
from reachy_nav.robot import BASE_Z                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gazebo_sdf_experiment import DEFAULT_URDF, truth_env          # noqa: E402

HOME = np.zeros(7)


def sample_goals(arm, truth, pairs, mobile, pool, n, rng, tries=4000):
    """Truth-verified, reachable goal configurations near the shelf."""
    qs, tips = pool
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        target = np.array([rng.uniform(0.30, 0.60),
                           rng.uniform(-0.45, 0.05),
                           rng.uniform(0.65, 1.35) + BASE_Z])
        order = np.argsort(np.linalg.norm(tips - target, axis=1))
        for k in order[:6]:
            q, err = arm.ik(target, seed=qs[k])
            if q is None:
                continue
            if worst_contact(arm, truth, q, pairs, mobile).clearance > 0.005:
                out.append((target, q))
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("out/shelf_map.npz"))
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--goals", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sensed = GridCache.load(args.map)
    truth = truth_env()
    print(f"map: {args.map} grid {tuple(int(d) for d in sensed.dims)}")

    client = p.connect(p.DIRECT)
    arm = ReachyArm(args.urdf, client, group="r_arm")
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    pool = arm.sample_pool(4000, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    goals = sample_goals(arm, truth, pairs, mobile, pool, args.goals, rng)
    print(f"{len(goals)} truth-verified collision-free goals sampled\n")

    def valid(q):
        return worst_contact(arm, sensed, q, pairs, mobile).clearance > 0.0

    refused = planned = collided = 0
    sl_collided = 0
    rows = []
    for i, (target, q) in enumerate(goals):
        mc = worst_contact(arm, sensed, q, pairs, mobile).clearance
        tc = worst_contact(arm, truth, q, pairs, mobile).clearance
        if mc <= 0.0:
            refused += 1
            rows.append((i, mc, tc, None, None, "REFUSED (free in truth)"))
            print(f"{i:2d} goal map {mc:+.4f} truth {tc:+.4f}  REFUSED by the map")
            continue
        t0 = time.monotonic()
        raw = pl.rrt_connect(HOME, q, arm.lo, arm.hi, valid,
                             timeout=args.timeout, seed=args.seed + i)
        if raw is None:
            rows.append((i, mc, tc, None, None, "planning failed"))
            print(f"{i:2d} goal map {mc:+.4f} truth {tc:+.4f}  planning FAILED "
                  f"({time.monotonic()-t0:.1f}s)")
            continue
        traj = pl.densify(pl.shortcut(raw, valid, seed=args.seed))
        lo_m = min(worst_contact(arm, sensed, qq, pairs, mobile).clearance
                   for qq in traj)
        worst_t = min((worst_contact(arm, truth, qq, pairs, mobile)
                       for qq in traj), key=lambda c: c.clearance)
        lo_t = worst_t.clearance
        planned += 1
        bad = lo_t <= 0.0
        collided += bad
        hit, sl_lo, _, _ = straight_line(arm, truth, HOME, q, pairs, mobile)
        sl_collided += bool(hit)
        rows.append((i, mc, tc, lo_m, lo_t,
                     "COLLIDES IN TRUTH" if bad else "safe"))
        print(f"{i:2d} goal map {mc:+.4f} truth {tc:+.4f} | path map {lo_m:+.4f} "
              f"truth {lo_t:+.4f} {'<-- COLLIDES' if bad else ''}"
              f"  [{worst_t.kind}: {worst_t.detail}]"
              f"  (straight line {'hits' if hit else 'clear'})")

    n = len(goals)
    print("\n" + "=" * 74)
    print(f"goals (all collision-free in truth):            {n}")
    print(f"  refused by the map though truth says free:    {refused} "
          f"({100*refused/max(n,1):.0f}%)   <- conservative")
    print(f"  planned:                                      {planned}")
    print(f"    of those, path collides in truth:           {collided} "
          f"({100*collided/max(planned,1):.0f}%)   <- dangerous")
    print(f"  straight-line baseline collides in truth:     {sl_collided}"
          f"/{planned}")
    if planned:
        worst = min(r[4] for r in rows if r[4] is not None)
        print(f"  worst true clearance on any planned path:     {worst:+.4f} m")
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
