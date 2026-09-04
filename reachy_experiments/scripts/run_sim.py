#!/usr/bin/env python3
"""Run the Reachy2 right-arm navigation scenes and report the two numbers.

    python scripts/run_sim.py --env shelf
    python scripts/run_sim.py --env all --headless
    python scripts/run_sim.py --env pillars --seed 3

For every goal it reports whether a collision-free configuration exists at the
goal pose (*reachable*) and whether the straight joint-space line from the home
configuration gets there without hitting anything. The gap between those two is
the planner-shaped hole.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_nav import ALL, ReachyArm, make                     # noqa: E402
from reachy_nav.motion import (movable_spheres, self_pairs,
                               straight_line, worst_contact)  # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")
HOME = np.zeros(7)
POOL_SIZE = 3000
IK_SEEDS = 12          # nearest pool configs tried per goal


def solve_goal(arm, env, goal, pairs, mobile, pool):
    """Best collision-free IK solution for a goal.

    Seeds come from the precomputed reachable pool, nearest tip first, rather
    than from random restarts: a DLS solve started from a configuration whose
    tip is already centimetres away converges in a few iterations, while one
    started at random usually does not converge at all.
    """
    qs, tips = pool
    order = np.argsort(np.linalg.norm(tips - np.asarray(goal.position), axis=1))
    best, best_err, best_clear, best_c = None, np.inf, -np.inf, None
    for k in order[:IK_SEEDS]:
        q, err = arm.ik(goal.position, seed=qs[k])
        if q is None:
            best_err = min(best_err, err)
            continue
        c = worst_contact(arm, env, q, pairs, mobile)
        if best is None or c.clearance > best_clear:
            best, best_err, best_clear, best_c = q, err, c.clearance, c
        if best_clear > 0.02:
            break
    return best, best_err, best_clear, best_c


def run_env(name, seed, urdf, gui):
    env = make(name, seed)
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    arm = ReachyArm(urdf, client)
    env.spawn(client)
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    pool = arm.sample_pool(POOL_SIZE, seed=seed)

    print(f"\n=== {name} ===")
    print(f"{env.description}  ({len(env.boxes)} obstacle bodies, "
          f"{len(env.goals)} goals, {len(mobile)}/{len(arm.spheres)} movable "
          f"spheres, {len(pairs)} self pairs checked)")

    reachable = collides = 0
    for gi, goal in enumerate(env.goals):
        q, err, clear, contact = solve_goal(arm, env, goal, pairs, mobile, pool)
        pos = np.asarray(goal.position)
        if q is None:
            print(f"  goal {gi} {np.round(pos,3)} ({goal.label}): "
                  f"UNREACHABLE - no IK within tol (best {err*1000:.1f} mm)")
            continue
        if clear <= 0.0:
            print(f"  goal {gi} {np.round(pos,3)} ({goal.label}): "
                  f"IK ok but goal config in collision "
                  f"(clearance {clear:+.4f} m, {contact.kind}: {contact.detail})")
            continue
        reachable += 1
        hit, lo, first, n = straight_line(arm, env, HOME, q, pairs, mobile)
        collides += bool(hit)
        verdict = (f"COLLISION at step {first}/{n}" if hit else "clear")
        print(f"  goal {gi} {np.round(pos,3)} ({goal.label}): reachable "
              f"(clearance {clear:+.4f} m [{contact.kind}: {contact.detail}], "
              f"IK err {err*1000:.1f} mm) | "
              f"straight line {verdict}, min clearance {lo:+.4f} m")

    n = len(env.goals)
    print(f"  -> {reachable}/{n} goal configs reachable & collision-free; "
          f"{collides}/{reachable} straight-line paths hit something")
    p.disconnect(client)
    return reachable, collides, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="all", help=f"one of {ALL} or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    gui = not args.headless and bool(os.environ.get("DISPLAY"))
    names = ALL if args.env == "all" else [args.env]

    tot_r = tot_c = tot_n = 0
    for name in names:
        r, c, n = run_env(name, args.seed, args.urdf, gui)
        tot_r += r; tot_c += c; tot_n += n
    print(f"\ntotal: {tot_r}/{tot_n} goals reachable & collision-free, "
          f"{tot_c}/{tot_r} of those unreachable by a straight joint-space line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
