#!/usr/bin/env python3
"""Plan against the live GP-SDF from Gazebo, then score the result against truth.

    python scripts/gazebo_sdf_experiment.py --video out/gazebo_sdf.mp4

Requires the reachy_sbc_experiment stack up on `shelf.world`:

    REACHY_WORLD=shelf.world docker compose --env-file config/scenarios/sim.env \\
        up -d --force-recreate reachy_sim
    docker compose --env-file config/scenarios/sim.env restart self_filter sdf_mapping

The point of the experiment is the gap between two collision models over the
*same* motion:

- **map**   -- `sdf_mapping`'s GP-SDF, built from the head ToF stream, which is
  what the planner is allowed to see;
- **truth** -- the analytic geometry of `shelf.world`, which nothing in the
  stack has access to.

The planner only ever consumes the map. Truth is used afterwards, to score.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_nav import ReachyArm                                   # noqa: E402
from reachy_nav import planner as pl                               # noqa: E402
from reachy_nav.live_sdf import GridCache, LiveSdfEnv              # noqa: E402
from reachy_nav.motion import (movable_spheres, self_pairs,        # noqa: E402
                               worst_contact)
from reachy_nav.robot import BASE_Z                                # noqa: E402
from reachy_nav.scene import Box, Env, Goal                        # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")
HOME = np.zeros(7)
W, H = 1280, 720
WOOD = (0.55, 0.35, 0.18, 1.0)


def truth_env() -> Env:
    """shelf.world, transcribed. odom z + BASE_Z gives the harness world."""
    def b(cx, cy, cz, hx, hy, hz, rgba=WOOD):
        return Box([cx, cy, cz + BASE_Z], [hx, hy, hz], rgba)
    boxes = [
        b(0.65, +0.45, 0.75, 0.175, 0.015, 0.75),        # side panels
        b(0.65, -0.45, 0.75, 0.175, 0.015, 0.75),
        b(0.65, 0.0, 0.60, 0.175, 0.465, 0.015),         # three shelves
        b(0.65, 0.0, 1.00, 0.175, 0.465, 0.015),
        b(0.65, 0.0, 1.40, 0.175, 0.465, 0.015),
        b(0.81, 0.0, 0.75, 0.015, 0.465, 0.75),          # back panel
        b(0.52, +0.12, 1.07, 0.04, 0.04, 0.05, (0.2, 0.5, 0.85, 1.0)),
        b(0.52, -0.12, 1.07, 0.04, 0.04, 0.05, (0.85, 0.25, 0.15, 1.0)),
    ]
    goals = [
        Goal([0.37, -0.12, 1.07 + BASE_Z], "+x", "standoff, 0.15 m from the right box"),
        Goal([0.44, -0.28, 0.82 + BASE_Z], "+x", "lower bay mouth, outboard"),
        Goal([0.44, -0.10, 1.22 + BASE_Z], "+x", "upper bay mouth, inboard"),
    ]
    return Env("shelf_world_truth", "shelf.world analytic geometry", boxes, goals)


class Rec:
    def __init__(self, out, fps=30):
        self.out, self.fps, self.n = Path(out), fps, 0
        self.dir = Path(tempfile.mkdtemp(prefix="reachy_gz_"))

    def frame(self, client, cap, sub, colour=(20, 110, 40)):
        view = p.computeViewMatrixFromYawPitchRoll([0.35, -0.15, 1.05], 2.1,
                                                   300, -14, 0, 2)
        proj = p.computeProjectionMatrixFOV(55, W / H, 0.05, 6.0)
        img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=client)[2]
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.asarray(img, dtype=np.uint8).reshape(H, W, 4),
                             "RGBA").convert("RGB")
        d = ImageDraw.Draw(im)
        d.text((24, 20), cap, fill=(15, 15, 15))
        d.text((24, 40), sub, fill=tuple(colour))
        d.text((24, 62), "geometry drawn is ground truth; the planner saw only "
                         "the GP-SDF built from the head ToF",
               fill=(90, 90, 90))
        im.save(self.dir / f"f{self.n:06d}.png")
        self.n += 1

    def hold(self, client, cap, sub, colour=(20, 110, 40), secs=0.9):
        for _ in range(max(1, int(secs * self.fps))):
            self.frame(client, cap, sub, colour)

    def close(self):
        if not self.n:
            return None
        self.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error",
                        "-framerate", str(self.fps),
                        "-i", str(self.dir / "f%06d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(self.out)],
                       check=True)
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resolution", type=float, default=0.04)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    truth = truth_env()
    live = LiveSdfEnv()
    t0 = time.time()
    sensed = GridCache(live, [-0.80, -1.00, 0.40], [0.95, 1.00, 2.00],
                       args.resolution)
    print(f"sensed map: grid {tuple(int(d) for d in sensed.dims)} = "
          f"{sensed.n_points} points in {time.time()-t0:.1f}s")

    client = p.connect(p.DIRECT if args.video else p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    arm = ReachyArm(args.urdf, client, group="r_arm")
    truth.spawn(client)
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    pool = arm.sample_pool(3000, seed=args.seed)
    rec = Rec(args.video, args.fps) if args.video else None
    if rec:
        rec.hold(client, "Gazebo GP-SDF experiment",
                 "planning against the sensed map, scored against shelf.world",
                 (20, 20, 20), 1.6)

    def valid(q):                      # what the planner is allowed to see
        return worst_contact(arm, sensed, q, pairs, mobile).clearance > 0.0

    rows = []
    for gi, goal in enumerate(truth.goals):
        qs, tips = pool
        order = np.argsort(np.linalg.norm(tips - np.asarray(goal.position), axis=1))
        q, mclear = None, -np.inf
        for k in order[:16]:
            cand, err = arm.ik(goal.position, seed=qs[k])
            if cand is None:
                continue
            c = worst_contact(arm, sensed, cand, pairs, mobile)
            if c.clearance > mclear:
                q, mclear = cand, c.clearance
            if mclear > 0.02:
                break
        if q is None or mclear <= 0.0:
            print(f"goal {gi} ({goal.label}): map says no free IK "
                  f"(best {mclear:+.4f} m)")
            rows.append((goal.label, None, None, None, None))
            if rec:
                rec.hold(client, f"goal {gi}: {goal.label}",
                         f"map refuses the goal (clearance {mclear:+.4f} m)",
                         (170, 30, 20), 1.0)
            continue
        tclear = worst_contact(arm, truth, q, pairs, mobile).clearance
        print(f"goal {gi} ({goal.label}): goal config map {mclear:+.4f} m, "
              f"truth {tclear:+.4f} m")

        raw = pl.rrt_connect(HOME, q, arm.lo, arm.hi, valid,
                             timeout=args.timeout, seed=args.seed)
        if raw is None:
            print("   planning failed")
            rows.append((goal.label, mclear, tclear, None, None))
            continue
        traj = pl.densify(pl.shortcut(raw, valid, seed=args.seed))
        lo_map = lo_truth = np.inf
        for k, qq in enumerate(traj):
            arm.set_config(qq)
            cm = worst_contact(arm, sensed, qq, pairs, mobile).clearance
            ct = worst_contact(arm, truth, qq, pairs, mobile).clearance
            lo_map, lo_truth = min(lo_map, cm), min(lo_truth, ct)
            if rec:
                bad = ct <= 0.0
                rec.frame(client, f"goal {gi}: {goal.label}",
                          f"step {k+1}/{len(traj)}   map {cm:+.3f} m   "
                          f"truth {ct:+.3f} m",
                          (170, 30, 20) if bad else (20, 110, 40))
        verdict = "SAFE" if lo_truth > 0 else "COLLIDES IN TRUTH"
        print(f"   path: {len(traj)} steps, min map {lo_map:+.4f} m, "
              f"min truth {lo_truth:+.4f} m  -> {verdict}")
        rows.append((goal.label, mclear, tclear, lo_map, lo_truth))
        if rec:
            rec.hold(client, f"goal {gi}: {goal.label}",
                     f"min clearance  map {lo_map:+.3f} m   truth {lo_truth:+.3f} m"
                     f"   -> {verdict}",
                     (20, 110, 40) if lo_truth > 0 else (170, 30, 20), 1.3)
        arm.zero()

    print("\n" + "=" * 78)
    print(f"{'goal':38s} {'goal map':>9s} {'goal truth':>10s} "
          f"{'path map':>9s} {'path truth':>10s}")
    for label, mc, tc, lm, lt in rows:
        f = lambda v: "  --  " if v is None else f"{v:+.4f}"
        print(f"{label:38s} {f(mc):>9s} {f(tc):>10s} {f(lm):>9s} {f(lt):>10s}")
    print(f"\nlive SDF: {live.stats()}")
    if rec:
        out = rec.close()
        if out:
            print(f"wrote {out} ({rec.n} frames, {rec.n/args.fps:.1f} s)")
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
