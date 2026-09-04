#!/usr/bin/env python3
"""Plan into the Gazebo clutter scene against the sensed map; score vs truth.

    python scripts/cache_sdf.py --out out/clutter_map.npz
    python scripts/clutter_experiment.py --map out/clutter_map.npz

Prints a goal joint vector that can be executed on the Gazebo robot with
scripts/e2e_group.py inside sbc_executor.
"""
from __future__ import annotations

import argparse
import sys
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
from reachy_nav.scene import Box, Env, Goal                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gazebo_sdf_experiment import DEFAULT_URDF                     # noqa: E402

HOME = np.zeros(7)

# clutter.world, transcribed. (name, centre xyz in odom, full size xyz)
OBSTACLES = [
    ("post_left",  (0.46, -0.02, 0.60), (0.16, 0.16, 1.20)),
    ("post_mid",   (0.52, -0.32, 0.70), (0.16, 0.16, 1.40)),
    ("block_low",  (0.56, -0.16, 0.92), (0.10, 0.10, 0.10)),
    ("target_object", (0.56, -0.16, 1.005), (0.07, 0.07, 0.07)),
]


def truth_env() -> Env:
    boxes = [Box([c[0], c[1], c[2] + BASE_Z], [s[0] / 2, s[1] / 2, s[2] / 2],
                 (0.45, 0.45, 0.5, 1.0)) for _, c, s in OBSTACLES]
    return Env("clutter_world_truth", "clutter.world analytic geometry", boxes, [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=Path, default=Path("out/clutter_map.npz"))
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--goals", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sensed = GridCache.load(args.map)
    truth = truth_env()
    client = p.connect(p.DIRECT)
    arm = ReachyArm(args.urdf, client, group="r_arm")
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    qs, tips = arm.sample_pool(5000, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    def valid(q):
        return worst_contact(arm, sensed, q, pairs, mobile).clearance > 0.0

    # goals in and behind the clutter, truth-verified free
    goals = []
    for _ in range(6000):
        if len(goals) >= args.goals:
            break
        t = np.array([rng.uniform(0.28, 0.55), rng.uniform(-0.52, 0.02),
                      rng.uniform(0.80, 1.30) + BASE_Z])
        for k in np.argsort(np.linalg.norm(tips - t, axis=1))[:6]:
            q, err = arm.ik(t, seed=qs[k])
            if q is None:
                continue
            if worst_contact(arm, truth, q, pairs, mobile).clearance > 0.020:
                goals.append((t, q))
                break
    print(f"{len(goals)} truth-verified goals in the clutter\n")

    refused = planned = collided = sl_hits = 0
    best_demo = None
    for i, (t, q) in enumerate(goals):
        mc = worst_contact(arm, sensed, q, pairs, mobile).clearance
        tc = worst_contact(arm, truth, q, pairs, mobile).clearance
        if mc <= 0:
            refused += 1
            print(f"{i:2d} {np.round(t,2)} map {mc:+.4f} truth {tc:+.4f}  REFUSED")
            continue
        raw = pl.rrt_connect(HOME, q, arm.lo, arm.hi, valid,
                             timeout=args.timeout, seed=args.seed + i)
        if raw is None:
            print(f"{i:2d} {np.round(t,2)} map {mc:+.4f} truth {tc:+.4f}  FAILED")
            continue
        traj = pl.densify(pl.shortcut(raw, valid, seed=args.seed))
        lm = min(worst_contact(arm, sensed, x, pairs, mobile).clearance for x in traj)
        wt = min((worst_contact(arm, truth, x, pairs, mobile) for x in traj),
                 key=lambda c: c.clearance)
        hit, _, _, _ = straight_line(arm, truth, HOME, q, pairs, mobile)
        sl_hits += bool(hit)
        planned += 1
        bad = wt.clearance <= 0
        collided += bad
        print(f"{i:2d} {np.round(t,2)} map {mc:+.4f} truth {tc:+.4f} | path map "
              f"{lm:+.4f} truth {wt.clearance:+.4f} "
              f"{'<-- COLLIDES [' + wt.detail + ']' if bad else 'safe'}"
              f"  (straight line {'hits' if hit else 'clear'})")
        if not bad and hit and (best_demo is None or wt.clearance > best_demo[1]):
            best_demo = (q, wt.clearance, t)     # safe, and needs a planner

    n = len(goals)
    print("\n" + "=" * 70)
    print(f"goals free in truth:                {n}")
    print(f"  refused by the map:               {refused}")
    print(f"  planned:                          {planned}")
    print(f"    collide in truth:               {collided}")
    print(f"  straight line collides:           {sl_hits}/{planned}")
    if best_demo is not None:
        q, cl, t = best_demo
        print(f"\nDEMO goal (safe in truth by {cl:+.4f} m, straight line collides), "
              f"tip odom {np.round(t - [0,0,BASE_Z],3).tolist()}")
        print("DEMOGOAL " + ",".join(f"{v:.4f}" for v in q))
        Path("/tmp/clutter_goal.txt").write_text(
            ",".join(f"{v:.4f}" for v in q))
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
