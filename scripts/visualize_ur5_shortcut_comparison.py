#!/usr/bin/env python3
"""Overlay and animate UR5 paths before and after shortcutting.

The inputs are the ``.path`` files written by ``demo_UR5PyBulletScene``.
Each file can contain one motion per scene goal; resets between motions are
handled by ``ur5_nav.load_runs`` and are never drawn as planned motion.

Example::

    python3 scripts/visualize_ur5_shortcut_comparison.py \
        --env corridor --before /tmp/corridor-before.path \
        --after /tmp/corridor-after.path --gui --hold

Orange is the unshortened path and blue is the shortcut path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ur5_experiments"))

from ur5_nav import SimSession, UR5, load_runs, make_env  # noqa: E402
from ur5_nav.envs import ENVIRONMENTS  # noqa: E402


BEFORE = (0.95, 0.35, 0.08, 0.48)
AFTER = (0.05, 0.65, 1.00, 0.72)
GOAL = (0.25, 1.00, 0.30)


def recolor(robot: UR5, rgba: tuple[float, float, float, float]) -> None:
    for link in robot.all_links:
        p.changeVisualShape(
            robot.body_id,
            link,
            rgbaColor=rgba,
            physicsClientId=robot.client,
        )


def tcp_trace(robot: UR5, run: np.ndarray) -> np.ndarray:
    points = []
    for q in run:
        robot.set_config(q)
        points.append(robot.ee_pose()[0])
    return np.asarray(points, dtype=float)


def arc_length(run: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())


def trace_markers(
    client: int,
    points: np.ndarray,
    rgba: tuple[float, float, float, float],
    limit: int = 220,
) -> None:
    if not len(points):
        return
    indices = np.linspace(0, len(points) - 1, min(limit, len(points))).round().astype(int)
    shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.007,
        rgbaColor=rgba,
        physicsClientId=client,
    )
    for point in points[indices]:
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=point,
            physicsClientId=client,
        )


def selected_runs(
    before: list[np.ndarray], after: list[np.ndarray], index: int | None
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    count = min(len(before), len(after))
    if index is not None:
        if index < 0 or index >= count:
            raise ValueError(f"--run must be between 0 and {count - 1}")
        return [(index, before[index], after[index])]
    return [(i, before[i], after[i]) for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=sorted(ENVIRONMENTS))
    parser.add_argument("--before", required=True, help="path generated with shortcut delta 0")
    parser.add_argument("--after", required=True, help="path generated with a positive shortcut delta")
    parser.add_argument("--run", type=int, help="show only this zero-based motion index")
    parser.add_argument("--gui", action="store_true", help="open the interactive PyBullet viewer")
    parser.add_argument("--hold", action="store_true", help="keep the GUI open after playback")
    parser.add_argument("--png", help="write a headless final-frame PNG")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.fps <= 0 or args.speed <= 0:
        parser.error("--fps and --speed must be positive")
    if not args.gui and not args.png:
        parser.error("select --gui, --png, or both")

    before_runs = load_runs(args.before)
    after_runs = load_runs(args.after)
    if len(before_runs) != len(after_runs):
        print(
            f"warning: {len(before_runs)} before motions and {len(after_runs)} after motions; "
            "comparing the shared prefix",
            file=sys.stderr,
        )
    try:
        runs = selected_runs(before_runs, after_runs, args.run)
    except ValueError as error:
        parser.error(str(error))

    with SimSession(gui=args.gui) as sim:
        env = make_env(args.env, sim.client, seed=0)
        before_robot = UR5(sim.client, self_collision=False)
        after_robot = UR5(sim.client, self_collision=False)
        recolor(before_robot, BEFORE)
        recolor(after_robot, AFTER)
        sim.set_camera(**env.camera)

        for goal in env.goals:
            sim.draw_marker(goal.position, rgb=GOAL, radius=0.025)

        traces: list[tuple[np.ndarray, np.ndarray]] = []
        for index, before, after in runs:
            before_trace = tcp_trace(before_robot, before)
            after_trace = tcp_trace(after_robot, after)
            traces.append((before_trace, after_trace))
            sim.draw_trace(before_trace, rgb=BEFORE[:3], width=3.0)
            sim.draw_trace(after_trace, rgb=AFTER[:3], width=5.0)
            if args.png:
                trace_markers(sim.client, before_trace, BEFORE)
                trace_markers(sim.client, after_trace, AFTER)
            old = arc_length(before)
            new = arc_length(after)
            change = 100.0 * (new / old - 1.0) if old else 0.0
            print(
                f"motion {index}: {old:.3f} -> {new:.3f} rad "
                f"({change:+.1f}%), {len(before)} -> {len(after)} states"
            )

        print("orange: shortcut disabled")
        print("blue:   shortcut enabled")

        if args.gui:
            delay = 1.0 / (args.fps * args.speed)
            for (_, before, after), _ in zip(runs, traces):
                count = max(len(before), len(after))
                for frame in range(count):
                    bi = round(frame * (len(before) - 1) / max(1, count - 1))
                    ai = round(frame * (len(after) - 1) / max(1, count - 1))
                    before_robot.set_config(before[bi])
                    after_robot.set_config(after[ai])
                    sim.step()
                    time.sleep(delay)

        # Leave both ghosts at the end of the last compared motion.
        _, before, after = runs[-1]
        before_robot.set_config(before[-1])
        after_robot.set_config(after[-1])
        sim.step()

        if args.png:
            sim.save_frame(args.png, **env.camera)
            print(f"wrote {args.png}")
        while args.gui and args.hold and p.isConnected(sim.client):
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
