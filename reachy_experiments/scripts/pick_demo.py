#!/usr/bin/env python3
"""Plan, reach into a shelf, grasp an object and carry it out. One arm.

    python scripts/pick_demo.py --video out/pick.mp4
    python scripts/pick_demo.py --hold          # watch it live

Each object is a real obstacle in the scene. For the object being picked it is
removed from the collision model -- you cannot plan a motion that touches the
thing you intend to grasp, and this stack has no allowed-collision mechanism,
so the target is excluded explicitly and every *other* object and the shelf
stay in the model. That is the honest version of "grasping" here: the reach and
the carry are planned and verified collision-free; the closing of the hand is
not simulated, the object is attached to the tip.
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
HOME = np.zeros(7)
W, H = 1280, 720
WOOD = (0.55, 0.35, 0.18, 1.0)
OBJ = [(0.85, 0.25, 0.15, 1.0), (0.20, 0.45, 0.85, 1.0), (0.95, 0.75, 0.15, 1.0)]

SHELF_X, DEPTH, WIDTH = 0.53, 0.12, 0.46
Y_C, BOTTOM, PITCH = -0.32, 0.80, 0.30
OBJ_HALF = np.array([0.033, 0.033, 0.050])


def build():
    """Shelf plus three graspable objects standing on its bay surfaces."""
    top = BOTTOM + 3 * PITCH
    boxes = [
        Box([SHELF_X + DEPTH, Y_C, top / 2], [0.02, WIDTH / 2, top / 2], WOOD),
        Box([SHELF_X, Y_C + WIDTH / 2, top / 2], [DEPTH, 0.02, top / 2], WOOD),
        Box([SHELF_X, Y_C - WIDTH / 2, top / 2], [DEPTH, 0.02, top / 2], WOOD),
    ]
    for k in range(3):
        boxes.append(Box([SHELF_X, Y_C, BOTTOM + k * PITCH],
                         [DEPTH, WIDTH / 2, 0.015], WOOD))
    n_shelf = len(boxes)
    objects = [
        ([SHELF_X - 0.01, Y_C + 0.11, BOTTOM + 0.015 + OBJ_HALF[2]], OBJ[0], "red can, lower bay"),
        ([SHELF_X - 0.01, Y_C - 0.10, BOTTOM + 0.015 + OBJ_HALF[2]], OBJ[1], "blue box, lower bay"),
        ([SHELF_X - 0.01, Y_C + 0.06, BOTTOM + PITCH + 0.015 + OBJ_HALF[2]], OBJ[2], "yellow box, middle bay"),
    ]
    for c, rgba, _ in objects:
        boxes.append(Box(list(c), OBJ_HALF.tolist(), rgba))
    goals = [Goal(c, "+x", label) for c, _, label in objects]
    env = Env("shelf_pick", "Shelf with three objects; pick each one out.",
              boxes, goals)
    return env, n_shelf


def without(env: Env, idx: int) -> Env:
    """The same scene with one object removed from the collision model."""
    return Env(env.name, env.description,
               [b for k, b in enumerate(env.boxes) if k != idx], env.goals)


class Rec:
    def __init__(self, out, fps=30):
        self.out, self.fps = Path(out), fps
        self.dir = Path(tempfile.mkdtemp(prefix="reachy_pick_"))
        self.n = 0

    def frame(self, client, cap, sub, colour=(20, 110, 40)):
        view = p.computeViewMatrixFromYawPitchRoll([0.28, -0.28, 1.05], 1.85,
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

    def hold(self, client, cap, sub, colour=(20, 110, 40), secs=0.7):
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
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", action="store_true")
    args = ap.parse_args()

    env, n_shelf = build()
    client = p.connect(p.DIRECT if args.video else p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.resetDebugVisualizerCamera(1.85, 300, -14, [0.28, -0.28, 1.05],
                                 physicsClientId=client)
    arm = ReachyArm(args.urdf, client)
    body_ids = env.spawn(client)
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    pool = arm.sample_pool(3000, seed=args.seed)
    rec = Rec(args.video, args.fps) if args.video else None

    if rec:
        rec.hold(client, "shelf_pick - three objects on the shelf",
                 "one arm (r_arm, 7 DoF), RRT-Connect, each reach verified "
                 "collision-free", (20, 20, 20), 1.5)

    for oi in range(3):
        bi = n_shelf + oi                      # index of this object's box
        goal = env.goals[oi]
        label = goal.label
        free = without(env, bi)                # target excluded, rest stay

        def valid(q, _e=free):
            return worst_contact(arm, _e, q, pairs, mobile).clearance > 0.0

        qs, tips = pool
        order = np.argsort(np.linalg.norm(tips - np.asarray(goal.position), axis=1))
        grasp, gclear = None, -np.inf
        for k in order[:16]:
            q, err = arm.ik(goal.position, seed=qs[k])
            if q is None:
                continue
            c = worst_contact(arm, free, q, pairs, mobile)
            if c.clearance > gclear:
                grasp, gclear = q, c.clearance
            if gclear > 0.01:
                break
        if grasp is None or gclear <= 0.0:
            print(f"object {oi} ({label}): no collision-free grasp configuration "
                  f"(best clearance {gclear:+.4f})")
            if rec:
                rec.hold(client, f"object {oi}: {label}",
                         "no reachable collision-free grasp", (170, 30, 20), 1.0)
            continue

        t0 = time.monotonic()
        raw = pl.rrt_connect(HOME, grasp, arm.lo, arm.hi, valid,
                             timeout=args.timeout, seed=args.seed)
        if raw is None:
            print(f"object {oi} ({label}): planning failed")
            if rec:
                rec.hold(client, f"object {oi}: {label}",
                         "RRT-Connect failed", (170, 30, 20), 1.0)
            continue
        reach = pl.densify(pl.shortcut(raw, valid, seed=args.seed))
        dt = time.monotonic() - t0
        print(f"object {oi} ({label}): reach planned in {dt:.2f}s, "
              f"{len(reach)} steps, grasp clearance {gclear:+.4f} m")

        lo = np.inf
        for k, q in enumerate(reach):
            arm.zero(); arm.set_config(q)
            lo = min(lo, worst_contact(arm, free, q, pairs, mobile).clearance)
            if rec:
                rec.frame(client, f"object {oi}: {label}",
                          f"REACH  step {k+1}/{len(reach)}  "
                          f"clearance {lo:+.3f} m")
            elif args.hold:
                time.sleep(0.012)
        print(f"    reach min clearance {lo:+.4f} m")
        if rec:
            rec.hold(client, f"object {oi}: {label}", "GRASP", (20, 110, 40), 0.8)

        # attach: the object rides with the tip from here on
        arm.zero(); arm.set_config(grasp)
        tip0 = arm.tip_position()
        obj_pos0, _ = p.getBasePositionAndOrientation(body_ids[bi],
                                                      physicsClientId=client)
        offset = np.asarray(obj_pos0) - tip0

        carry_raw = pl.rrt_connect(grasp, HOME, arm.lo, arm.hi, valid,
                                   timeout=args.timeout, seed=args.seed + 1)
        carry = (pl.densify(pl.shortcut(carry_raw, valid, seed=args.seed))
                 if carry_raw is not None else list(reversed(reach)))
        for k, q in enumerate(carry):
            arm.zero(); arm.set_config(q)
            p.resetBasePositionAndOrientation(
                body_ids[bi], (arm.tip_position() + offset).tolist(),
                [0, 0, 0, 1], physicsClientId=client)
            if rec:
                rec.frame(client, f"object {oi}: {label}",
                          f"CARRY OUT  step {k+1}/{len(carry)}  object attached")
            elif args.hold:
                time.sleep(0.012)
        print(f"    carried out in {len(carry)} steps")
        # park the object clear of the shelf so the next pick is unobstructed
        p.resetBasePositionAndOrientation(
            body_ids[bi], [0.18, -0.62 - 0.10 * oi, 0.10],
            [0, 0, 0, 1], physicsClientId=client)
        arm.zero()
        if rec:
            rec.hold(client, f"object {oi}: {label}", "PLACED", (20, 110, 40), 0.6)

    if rec:
        out = rec.close()
        if out:
            print(f"\nwrote {out} ({rec.n} frames, {rec.n/args.fps:.1f} s)")
    if args.hold and not args.video:
        print("window open; Ctrl-C to exit")
        try:
            while p.isConnected(physicsClientId=client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
