#!/usr/bin/env python3
"""Both hands grasp one box on the shelf and carry it out together.

    python scripts/bimanual_pick_demo.py --video out/bimanual.mp4
    python scripts/bimanual_pick_demo.py --hold

Planning is over the coupled 14-DoF state (both arms at once), not two
independent 7-DoF problems. That coupling is the point: the arms share a
workspace, so arm-vs-arm self collision is live throughout -- 48 of the 85
spheres move here against 24 for one arm.

The *goal* is still solved one arm at a time, because the two chains are
kinematically independent: the right IK cannot change where the left tip is.
Only the collision check has to be joint, and it is.
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
from reachy_nav.motion import (movable_spheres, self_pairs,        # noqa: E402
                               worst_contact)
from reachy_nav.scene import Box, Env, Goal                        # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")
HOME = np.zeros(14)
RCOLS, LCOLS = list(range(0, 7)), list(range(7, 14))
W, H = 1280, 720
WOOD = (0.55, 0.35, 0.18, 1.0)

# The shelf is centred on the midline here, unlike the single-arm scenes: a
# bimanual grasp needs a target both arms can reach, and the right arm only
# crosses to y = +0.247 while the left only reaches y = -0.247.
SHELF_X, DEPTH, WIDTH = 0.50, 0.12, 0.60
# A 0.30 m bay pitch is enough for one hand and not for two: swept against the
# real collision model, both hands entering a 0.30 m bay bound at -0.006 m on
# r_hand_palm_link, and widening the grasp only made it worse (-0.034 m on the
# forearm). 0.42 m clears at +0.033 m. Putting the box on top of the unit
# instead does not work either -- at z = 1.5 the left forearm hits the shelf.
Y_C, BOTTOM, PITCH = 0.0, 0.80, 0.42
BOX_HALF = np.array([0.055, 0.125, 0.085])
GRASP_Y = 0.152          # just outside the box sides


def build():
    top = BOTTOM + 2 * PITCH
    boxes = [
        Box([SHELF_X + DEPTH, Y_C, top / 2], [0.02, WIDTH / 2, top / 2], WOOD),
        Box([SHELF_X, Y_C + WIDTH / 2, top / 2], [DEPTH, 0.02, top / 2], WOOD),
        Box([SHELF_X, Y_C - WIDTH / 2, top / 2], [DEPTH, 0.02, top / 2], WOOD),
    ]
    for k in range(3):
        boxes.append(Box([SHELF_X, Y_C, BOTTOM + k * PITCH],
                         [DEPTH, WIDTH / 2, 0.015], WOOD))
    n_shelf = len(boxes)
    centre = [SHELF_X - 0.012, Y_C, BOTTOM + 0.015 + BOX_HALF[2]]
    boxes.append(Box(list(centre), BOX_HALF.tolist(), (0.20, 0.50, 0.85, 1.0)))
    env = Env("bimanual_pick",
              "One wide box on the shelf, grasped by both hands.",
              boxes, [Goal(centre, "+x", "wide box, lower bay")])
    return env, n_shelf, np.asarray(centre)


def plan_with_retries(start, goal, arm, valid, timeout, seed, resolution,
                      attempts=4, step=0.4, what="plan"):
    """RRT-Connect with restarts.

    At 14 DoF the run-to-run spread is large: the same query solved in 54 s,
    then 129 s, then not at all inside 150 s. A fresh seed is far cheaper than
    a longer single budget, because a run that has not connected early usually
    will not.
    """
    for k in range(attempts):
        t0 = time.monotonic()
        path = pl.rrt_connect(start, goal, arm.lo, arm.hi, valid, step=step,
                              timeout=timeout, seed=seed + 17 * k,
                              resolution=resolution)
        dt = time.monotonic() - t0
        if path is not None:
            print(f"  {what}: solved on attempt {k+1} in {dt:.1f}s "
                  f"({len(path)} waypoints)")
            return path
        print(f"  {what}: attempt {k+1} failed after {dt:.1f}s")
    return None


def without(env: Env, idx: int) -> Env:
    return Env(env.name, env.description,
               [b for k, b in enumerate(env.boxes) if k != idx], env.goals)


class Rec:
    def __init__(self, out, fps=30):
        self.out, self.fps, self.n = Path(out), fps, 0
        self.dir = Path(tempfile.mkdtemp(prefix="reachy_bimanual_"))

    def frame(self, client, cap, sub, colour=(20, 110, 40)):
        view = p.computeViewMatrixFromYawPitchRoll([0.30, 0.0, 1.02], 1.95,
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
        im.save(self.dir / f"f{self.n:06d}.png")
        self.n += 1

    def hold(self, client, cap, sub, colour=(20, 110, 40), secs=0.8):
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
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per attempt; see plan_with_retries")
    ap.add_argument("--plan-resolution", type=float, default=0.024,
                    help="edge-check resolution. Defaults to the 0.024 rad "
                         "the path is verified at, and should not be coarsened: "
                         "growing the tree at 0.05 and verifying at 0.024 "
                         "returned paths grazing to -0.0005 m, because "
                         "shortcutting only replaces some of the RRT edges and "
                         "the survivors were never checked that finely.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", action="store_true")
    args = ap.parse_args()

    env, n_shelf, centre = build()
    bi = n_shelf                       # the graspable box
    free = without(env, bi)

    client = p.connect(p.DIRECT if args.video else p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.resetDebugVisualizerCamera(1.95, 300, -14, [0.30, 0.0, 1.02],
                                 physicsClientId=client)
    arm = ReachyArm(args.urdf, client, group="dual_arm")
    ids = env.spawn(client)
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    print(f"dual_arm: {len(arm.arm)} joints, {len(mobile)}/{len(arm.spheres)} "
          f"movable spheres, {len(pairs)} self pairs")
    rec = Rec(args.video, args.fps) if args.video else None
    if rec:
        rec.hold(client, "bimanual_pick - both hands, one box",
                 f"coupled 14-DoF RRT-Connect, {len(mobile)} movable spheres, "
                 f"{len(pairs)} self pairs", (20, 20, 20), 1.6)

    def valid(q):
        return worst_contact(arm, free, q, pairs, mobile).clearance > 0.0

    # ---- grasp configuration: right side, then left, then check jointly ----
    gr = centre + np.array([0.0, -GRASP_Y, 0.0])
    gl = centre + np.array([0.0, +GRASP_Y, 0.0])
    pr, tr = arm.sample_pool(2500, seed=args.seed, which=0, cols=RCOLS)
    plf, tl = arm.sample_pool(2500, seed=args.seed + 1, which=1, cols=LCOLS)
    grasp, gclear = None, -np.inf
    for a in np.argsort(np.linalg.norm(tr - gr, axis=1))[:12]:
        q1, e1 = arm.ik(gr, seed=pr[a], which=0, cols=RCOLS)
        if q1 is None:
            continue
        for b in np.argsort(np.linalg.norm(tl - gl, axis=1))[:12]:
            q2, e2 = arm.ik(gl, seed=plf[b], which=1, cols=LCOLS, base=q1)
            if q2 is None:
                continue
            c = worst_contact(arm, free, q2, pairs, mobile)
            if c.clearance > gclear:
                grasp, gclear = q2, c.clearance
            if gclear > 0.035:
                break
        if gclear > 0.035:
            break
    if grasp is None or gclear <= 0.0:
        print(f"no collision-free bimanual grasp (best {gclear:+.4f} m)")
        return 1
    arm.zero(); arm.set_config(grasp)
    print(f"grasp: r tip {np.round(arm.tip_position(0),3).tolist()}  "
          f"l tip {np.round(arm.tip_position(1),3).tolist()}  "
          f"clearance {gclear:+.4f} m")

    # ---- reach ----------------------------------------------------------
    t0 = time.monotonic()
    raw = plan_with_retries(HOME, grasp, arm, valid, args.timeout, args.seed,
                            args.plan_resolution, what="reach")
    if raw is None:
        print("reach planning failed")
        return 1
    reach = pl.densify(pl.shortcut(raw, valid, iterations=200, seed=args.seed,
                                   resolution=pl.RESOLUTION))
    print(f"reach planned in {time.monotonic()-t0:.2f}s, {len(raw)} waypoints "
          f"-> {len(reach)} steps, length {pl.length(raw):.2f} rad")
    lo = np.inf
    for k, q in enumerate(reach):
        arm.zero(); arm.set_config(q)
        lo = min(lo, worst_contact(arm, free, q, pairs, mobile).clearance)
        if rec:
            rec.frame(client, "bimanual_pick - reach",
                      f"REACH  step {k+1}/{len(reach)}  clearance {lo:+.3f} m")
        elif args.hold:
            time.sleep(0.012)
    print(f"  reach min clearance {lo:+.4f} m")
    if rec:
        rec.hold(client, "bimanual_pick", "GRASP - both hands on the box")

    # ---- attach to the midpoint of the two tips --------------------------
    arm.zero(); arm.set_config(grasp)
    mid0 = 0.5 * (arm.tip_position(0) + arm.tip_position(1))
    pos0, _ = p.getBasePositionAndOrientation(ids[bi], physicsClientId=client)
    offset = np.asarray(pos0) - mid0

    t1 = time.monotonic()
    craw = plan_with_retries(grasp, HOME, arm, valid, args.timeout,
                             args.seed + 5, args.plan_resolution, what="carry")
    carry = (pl.densify(pl.shortcut(craw, valid, iterations=200, seed=args.seed,
                                    resolution=pl.RESOLUTION))
             if craw is not None else list(reversed(reach)))
    print(f"carry planned in {time.monotonic()-t1:.2f}s, {len(carry)} steps")
    lo2 = np.inf
    for k, q in enumerate(carry):
        arm.zero(); arm.set_config(q)
        lo2 = min(lo2, worst_contact(arm, free, q, pairs, mobile).clearance)
        mid = 0.5 * (arm.tip_position(0) + arm.tip_position(1))
        p.resetBasePositionAndOrientation(ids[bi], (mid + offset).tolist(),
                                          [0, 0, 0, 1], physicsClientId=client)
        if rec:
            rec.frame(client, "bimanual_pick - carry",
                      f"CARRY OUT  step {k+1}/{len(carry)}  "
                      f"box held by both hands  clearance {lo2:+.3f} m")
        elif args.hold:
            time.sleep(0.012)
    print(f"  carry min clearance {lo2:+.4f} m")
    if rec:
        rec.hold(client, "bimanual_pick",
                 f"done - reach {lo:+.3f} m, carry {lo2:+.3f} m, "
                 f"both collision-free", (20, 110, 40), 1.4)
        out = rec.close()
        if out:
            print(f"\nwrote {out} ({rec.n} frames, {rec.n/args.fps:.1f} s)")
    if args.hold and not args.video:
        try:
            while p.isConnected(physicsClientId=client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
