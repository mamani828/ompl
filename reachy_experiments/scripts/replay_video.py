#!/usr/bin/env python3
"""Replay a planned .path from the C++ Reachy2 demos and record it.

Unlike ``demo.py`` -- which replays the naive straight line to show that it
fails -- this animates a trajectory the CBF planner actually produced, so the
arm routes *around* the shelf instead of through it.

    ./build/demos/demo_Reachy2CBFPlanning 10 reachy2_cbf.path 0.05 5
    python scripts/replay_video.py reachy2_cbf.path --video planned.mp4

The scene is the built-in shelf from ``demos/Reachy2CBFPlanningDemo.cpp``,
reproduced here so the picture matches what the planner was solving against.
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
from reachy_nav.robot import BASE_Z, spheres_only_urdf          # noqa: E402
from reachy_nav.scene import Box                                # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")
W, H = 1280, 720


def demo_shelf():
    """The shelf from Reachy2CBFPlanningDemo.cpp, verbatim."""
    bottom, shelf_x, depth, width = 0.9144, 0.62, 0.14, 0.75
    pitch, panel_half = 0.44, 0.46
    top = bottom + 2 * panel_half
    boxes = [
        Box([shelf_x + depth, 0, top / 2], [0.02, width / 2, top / 2]),
        Box([shelf_x, width / 2, top / 2], [depth, 0.02, top / 2]),
        Box([shelf_x, -width / 2, top / 2], [depth, 0.02, top / 2]),
    ]
    for z in (0.0, pitch, 2 * pitch):
        boxes.append(Box([shelf_x, 0, bottom + z], [depth, width / 2, 0.015]))
    return boxes


def read_path(path: Path):
    names, rows = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.lstrip("#").split()
            if parts and parts[0] == "joints":
                names = parts[1:]
            continue
        rows.append([float(v) for v in line.split()])
    return names, np.asarray(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=2, help="loops of the motion")
    ap.add_argument("--caption", default="")
    args = ap.parse_args()

    names, rows = read_path(args.path)
    if rows.size == 0:
        print("no configurations in", args.path)
        return 1
    # the mobile format prepends base x/y/yaw
    base_cols = max(0, rows.shape[1] - len(names))
    print(f"{len(rows)} waypoints, {len(names)} joints"
          + (f", {base_cols} base DoF" if base_cols else ""))

    client = p.connect(p.DIRECT if args.video else p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    tmp = tempfile.TemporaryDirectory()
    staged = Path(tmp.name) / "spheres.urdf"
    spheres_only_urdf(args.urdf, staged)
    body = p.loadURDF(str(staged), basePosition=[0, 0, BASE_Z],
                      useFixedBase=True, physicsClientId=client)
    idx = {}
    for i in range(p.getNumJoints(body, physicsClientId=client)):
        idx[p.getJointInfo(body, i, physicsClientId=client)[1].decode()] = i
    cols = [idx[n] for n in names]

    for b in demo_shelf():
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(b.half),
                                     physicsClientId=client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(b.half),
                                  rgbaColor=list(b.rgba), physicsClientId=client)
        p.createMultiBody(0, col, vis, list(b.center), physicsClientId=client)

    frames = Path(tempfile.mkdtemp(prefix="reachy_replay_")) if args.video else None
    cam = [0.30, 0.0, 1.15]
    n = 0
    caption = args.caption or f"CBF-planned trajectory - {args.path.name}"
    for _ in range(max(1, args.repeat)):
        for r, row in enumerate(rows):
            base = row[:base_cols]
            q = row[base_cols:]
            if base_cols >= 3:
                yaw = float(base[2])
                p.resetBasePositionAndOrientation(
                    body, [float(base[0]), float(base[1]), BASE_Z],
                    p.getQuaternionFromEuler([0, 0, yaw]),
                    physicsClientId=client)
            for j, v in zip(cols, q):
                p.resetJointState(body, j, float(v), physicsClientId=client)
            if frames is not None:
                view = p.computeViewMatrixFromYawPitchRoll(cam, 2.3, 300, -14, 0, 2)
                proj = p.computeProjectionMatrixFOV(55, W / H, 0.05, 6.0)
                img = p.getCameraImage(W, H, view, proj,
                                       renderer=p.ER_TINY_RENDERER,
                                       physicsClientId=client)[2]
                from PIL import Image, ImageDraw
                im = Image.fromarray(np.asarray(img, dtype=np.uint8)
                                     .reshape(H, W, 4), "RGBA").convert("RGB")
                d = ImageDraw.Draw(im)
                d.text((24, 20), caption, fill=(15, 15, 15))
                d.text((24, 40), f"waypoint {r+1}/{len(rows)} - "
                                 f"collision-free by construction",
                       fill=(20, 110, 40))
                im.save(frames / f"f{n:06d}.png")
                n += 1
            else:
                time.sleep(1.0 / args.fps)

    if frames is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error",
                        "-framerate", str(args.fps), "-i", str(frames / "f%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(args.video)], check=True)
        shutil.rmtree(frames, ignore_errors=True)
        print(f"wrote {args.video} ({n} frames, {n/args.fps:.1f} s)")
    p.disconnect(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
