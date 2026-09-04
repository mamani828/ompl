#!/usr/bin/env python3
"""Watch Reachy2's right arm attempt each scene.

    python scripts/demo.py --env shelf
    python scripts/demo.py --env all --speed 2
    python scripts/demo.py --env corridor --png corridor.png

Green spheres are goals. For every goal the arm replays the *straight
joint-space line* from the home configuration -- the naive plan -- and the
spheres turn red on any step where a collision-model sphere is in contact.
Seeing where it turns red is the whole point of the scene: that is the motion a
planner has to replace.
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
from reachy_nav import ALL, ReachyArm, make                        # noqa: E402
from reachy_nav.motion import (movable_spheres, self_pairs,        # noqa: E402
                               worst_contact)
from reachy_nav import planner as pl                               # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")
HOME = np.zeros(7)
POOL, SEEDS = 3000, 12
OK = (0.15, 0.80, 0.25, 1.0)
BAD = (0.90, 0.15, 0.10, 1.0)


VIDEO_W, VIDEO_H = 1280, 720


class Recorder:
    """Offscreen frame capture, encoded to mp4 by ffmpeg at the end.

    Rendered with ``getCameraImage`` rather than grabbed off the screen: the
    frames are deterministic, independent of window size and of whatever else
    is on the desktop, and it works headless.
    """

    def __init__(self, out: Path, fps: int = 30):
        self.out = Path(out)
        self.fps = fps
        self.dir = Path(tempfile.mkdtemp(prefix="reachy_demo_"))
        self.n = 0

    def frame(self, client, target, caption, subcaption="", colour=(20, 20, 20)):
        view = p.computeViewMatrixFromYawPitchRoll(target, 2.0, 300, -16, 0, 2)
        proj = p.computeProjectionMatrixFOV(55, VIDEO_W / VIDEO_H, 0.05, 6.0)
        img = p.getCameraImage(VIDEO_W, VIDEO_H, view, proj,
                               renderer=p.ER_TINY_RENDERER,
                               physicsClientId=client)[2]
        from PIL import Image, ImageDraw
        im = Image.fromarray(np.asarray(img, dtype=np.uint8)
                             .reshape(VIDEO_H, VIDEO_W, 4), "RGBA").convert("RGB")
        d = ImageDraw.Draw(im)
        d.text((24, 20), caption, fill=(15, 15, 15))
        if subcaption:
            d.text((24, 40), subcaption, fill=tuple(colour))
        im.save(self.dir / f"f{self.n:06d}.png")
        self.n += 1

    def hold(self, client, target, caption, subcaption="", colour=(20, 20, 20),
             seconds: float = 0.6):
        for _ in range(max(1, int(seconds * self.fps))):
            self.frame(client, target, caption, subcaption, colour)

    def close(self):
        if self.n == 0:
            return None
        self.out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg not found; frames left in", self.dir)
            return None
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(self.fps),
               "-i", str(self.dir / "f%06d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "20", str(self.out)]
        subprocess.run(cmd, check=True)
        shutil.rmtree(self.dir, ignore_errors=True)
        return self.out


def marker(client, pos, rgba, radius=0.022):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba),
                              physicsClientId=client)
    return p.createMultiBody(0, -1, vis, list(pos), physicsClientId=client)


def solve(arm, env, goal, pairs, mobile, pool):
    qs, tips = pool
    order = np.argsort(np.linalg.norm(tips - np.asarray(goal.position), axis=1))
    best, best_clear = None, -np.inf
    for k in order[:SEEDS]:
        q, err = arm.ik(goal.position, seed=qs[k])
        if q is None:
            continue
        c = worst_contact(arm, env, q, pairs, mobile)
        if best is None or c.clearance > best_clear:
            best, best_clear = q, c.clearance
        if best_clear > 0.02:
            break
    return best, best_clear


def run(name, seed, urdf, client, speed, png, rec=None, plan=False,
        timeout=20.0):
    env = make(name, seed)
    p.resetSimulation(physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    arm = ReachyArm(urdf, client)
    env.spawn(client)
    pairs = self_pairs(arm, HOME)
    mobile = movable_spheres(arm)
    pool = arm.sample_pool(POOL, seed=seed)

    handles = [marker(client, g.position, OK) for g in env.goals]
    cam = [0.30, -0.28, 1.12]
    if rec is not None:
        rec.hold(client, cam, f"{name} - {env.description}",
                 (f"{len(env.goals)} goals, RRT-Connect" if plan
                  else f"{len(env.goals)} goals, straight-line joint interpolation"),
                 (20, 20, 20), 1.2)
    p.addUserDebugText(f"{name}: {env.description}", [0.0, -0.35, 1.85],
                       textColorRGB=[0.1, 0.1, 0.1], textSize=1.3,
                       physicsClientId=client)
    print(f"\n=== {name} === {env.description}")

    for gi, goal in enumerate(env.goals):
        q, clear = solve(arm, env, goal, pairs, mobile, pool)
        if q is None:
            print(f"  goal {gi} ({goal.label}): unreachable")
            p.changeVisualShape(handles[gi], -1, rgbaColor=list(BAD),
                                physicsClientId=client)
            if rec is not None:
                rec.hold(client, cam, f"{name} - goal {gi}: {goal.label}",
                         "UNREACHABLE - no IK within tolerance", (170, 30, 20), 0.8)
            continue
        if plan:
            def valid(qq, _a=arm, _e=env, _p=pairs, _m=mobile):
                return worst_contact(_a, _e, qq, _p, _m).clearance > 0.0
            t0 = time.monotonic()
            raw = pl.rrt_connect(HOME, q, arm.lo, arm.hi, valid,
                                 timeout=timeout, seed=seed)
            dt = time.monotonic() - t0
            if raw is None:
                print(f"  goal {gi} ({goal.label}): PLANNING FAILED in {dt:.1f}s")
                p.changeVisualShape(handles[gi], -1, rgbaColor=list(BAD),
                                    physicsClientId=client)
                if rec is not None:
                    rec.hold(client, cam, f"{name} - goal {gi}: {goal.label}",
                             f"RRT-Connect FAILED after {dt:.1f} s",
                             (170, 30, 20), 0.8)
                continue
            short = pl.shortcut(raw, valid, seed=seed)
            traj = pl.densify(short)
            print(f"  goal {gi} ({goal.label}): planned in {dt:.2f}s, "
                  f"{len(raw)} waypoints -> {len(short)} after shortcut, "
                  f"length {pl.length(raw):.2f} -> {pl.length(short):.2f} rad, "
                  f"{len(traj)} steps")
            lo_clear = np.inf
            for k, qk in enumerate(traj):
                arm.zero(); arm.set_config(qk)
                c = worst_contact(arm, env, qk, pairs, mobile)
                lo_clear = min(lo_clear, c.clearance)
                if rec is not None:
                    rec.frame(client, cam, f"{name} - goal {gi}: {goal.label}",
                              f"RRT-Connect plan, step {k+1}/{len(traj)}, "
                              f"clearance {c.clearance:+.3f} m", (20, 110, 40))
                else:
                    time.sleep(max(0.0, 0.012 / max(speed, 1e-6)))
            ok = lo_clear > 0.0
            print(f"      min clearance along the planned path "
                  f"{lo_clear:+.4f} m -> {'COLLISION-FREE' if ok else 'INVALID'}")
            if rec is not None:
                rec.hold(client, cam, f"{name} - goal {gi}: {goal.label}",
                         f"planned path is collision-free "
                         f"(min clearance {lo_clear:+.3f} m)"
                         if ok else "planned path is INVALID",
                         (20, 110, 40) if ok else (170, 30, 20), 0.8)
            arm.zero()
            continue

        span = float(np.max(np.abs(q - HOME)))
        steps = max(2, int(np.ceil(span / 0.024)) + 1)
        hit = False
        for k in range(steps):
            qk = HOME + (q - HOME) * (k / (steps - 1))
            arm.zero(); arm.set_config(qk)
            c = worst_contact(arm, env, qk, pairs, mobile)
            if c.clearance < 0.0 and not hit:
                hit = True
                p.changeVisualShape(handles[gi], -1, rgbaColor=list(BAD),
                                    physicsClientId=client)
                print(f"  goal {gi} ({goal.label}): straight line COLLIDES at "
                      f"step {k}/{steps} ({c.kind}: {c.detail})")
            if rec is not None:
                sub = ("straight line COLLIDES" if hit
                       else f"clearance {c.clearance:+.3f} m")
                rec.frame(client, cam, f"{name} - goal {gi}: {goal.label}",
                          sub, (170, 30, 20) if hit else (20, 110, 40))
            else:
                time.sleep(max(0.0, 0.012 / max(speed, 1e-6)))
        if not hit:
            print(f"  goal {gi} ({goal.label}): straight line is clear")
        if rec is not None:
            rec.hold(client, cam, f"{name} - goal {gi}: {goal.label}",
                     "straight line COLLIDES - a planner must replace this motion"
                     if hit else "straight line is clear",
                     (170, 30, 20) if hit else (20, 110, 40), 0.7)
        else:
            time.sleep(0.4 / max(speed, 1e-6))
        arm.zero()
    if png:
        w, h = 1280, 900
        view = p.computeViewMatrixFromYawPitchRoll((0.30, -0.28, 1.12), 2.0,
                                                   300, -16, 0, 2)
        proj = p.computeProjectionMatrixFOV(55, w / h, 0.05, 6.0)
        img = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=client)[2]
        from PIL import Image
        Image.fromarray(np.asarray(img, dtype=np.uint8).reshape(h, w, 4),
                        "RGBA").convert("RGB").save(png)
        print(f"  saved {png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="shelf", help=f"one of {ALL} or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--png", type=Path, help="save a frame per scene (adds a suffix for 'all')")
    ap.add_argument("--hold", action="store_true", help="keep the window open at the end")
    ap.add_argument("--video", type=Path,
                    help="record an mp4 (forces headless offscreen rendering)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--plan", action="store_true",
                    help="plan with RRT-Connect instead of driving the straight line")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    headless = args.headless or args.video is not None
    client = p.connect(p.DIRECT if headless else p.GUI)
    rec = Recorder(args.video, args.fps) if args.video else None
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.resetDebugVisualizerCamera(1.9, 300, -16, [0.30, -0.28, 1.12],
                                 physicsClientId=client)
    names = ALL if args.env == "all" else [args.env]
    for name in names:
        png = None
        if args.png:
            png = (args.png if len(names) == 1
                   else args.png.with_name(f"{args.png.stem}_{name}{args.png.suffix}"))
        run(name, args.seed, args.urdf, client, args.speed, png, rec,
            args.plan, args.timeout)
    if rec is not None:
        out = rec.close()
        if out:
            print(f"\nwrote {out} ({rec.n} frames at {args.fps} fps, "
                  f"{rec.n/args.fps:.1f} s)")
    if args.hold and not headless:
        print("\nwindow open; Ctrl-C to exit")
        try:
            while p.isConnected(physicsClientId=client):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
