#!/usr/bin/env python3
"""Replay a Reachy2 CBF path with the exact shelf used by the OMPL demo.

Examples:
  python3 scripts/visualize_reachy2_cbf.py reachy2_cbf.path --gui --hold
  python3 scripts/visualize_reachy2_cbf.py reachy2_cbf.path --png reachy2.png

The supplied Reachy URDF contains visual mesh paths rooted at another user's
home directory. By default this tool creates a temporary visualization URDF in
which the spherized collision geometry is also the visual geometry, so it works
without those missing meshes. Pass --meshed-urdf when its mesh paths are valid.
"""
from __future__ import annotations

import argparse
import copy
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p


DEFAULT_URDF = Path.home() / "safe_bubble_cover/robots/reachy2/reachy2_spherized.urdf"
WOOD = (0.62, 0.44, 0.26, 1.0)
# The URDF's mobile-base visual is authored 0.1075 m below base_link.  Spawn
# base_link at that height so the physical base, rather than its frame origin,
# rests on z=0.  This must match Reachy2::groundedBasePose().
GROUNDED_BASE_Z = 0.1075


def read_path(path: Path):
    with path.open() as stream:
        header = stream.readline().strip().split()
    if header[:2] != ["#", "joints"]:
        raise ValueError(f"{path}: expected '# joints ...' header")
    names = header[2:]
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.shape[1] != len(names):
        raise ValueError(f"{path}: header has {len(names)} joints, rows have {values.shape[1]}")
    return names, values


def sphere_visual_urdf(source: Path, destination: Path):
    """Make collision spheres visible and discard unavailable mesh visuals."""
    tree = ET.parse(source)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for collision in link.findall("collision"):
            if collision.find("geometry/sphere") is None:
                continue
            visual = ET.Element("visual")
            origin = collision.find("origin")
            if origin is not None:
                visual.append(copy.deepcopy(origin))
            visual.append(copy.deepcopy(collision.find("geometry")))
            side = link.get("name", "")
            material = ET.SubElement(visual, "material", {"name": f"cbf_sphere_{side}"})
            rgba = "0.15 0.45 0.95 0.78" if side.startswith("l_") else \
                   "0.95 0.25 0.18 0.78" if side.startswith("r_") else \
                   "0.55 0.58 0.62 0.48"
            ET.SubElement(material, "color", {"rgba": rgba})
            link.append(visual)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def add_box(half, center, rgba=WOOD):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(0, collision, visual, basePosition=center)


def add_shelf():
    # Floor-standing version of the UR5 shelf.  Its mounting table is not part
    # of Reachy's scene; the back and side boards support the shelf from z=0.
    bottom, sx, depth, width, pitch, original_panel_half = (
        0.9144, 0.62, 0.14, 0.75, 0.44, 0.46
    )
    top = bottom + 2*original_panel_half
    add_box((0.02, width/2, top/2), (sx+depth, 0, top/2))
    add_box((depth, 0.02, top/2), (sx, width/2, top/2))
    add_box((depth, 0.02, top/2), (sx, -width/2, top/2))
    for z in (0, pitch, 2*pitch):
        add_box((depth, width/2, 0.015), (sx, 0, bottom+z))


def add_ground():
    add_box((2.0, 2.0, 0.01), (0.0, 0.0, -0.01), (0.72, 0.72, 0.70, 1.0))


def add_marker(position, rgba):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=rgba)
    p.createMultiBody(0, -1, visual, basePosition=position)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="path written by demo_Reachy2CBFPlanning")
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF,
                    help=f"spherized Reachy URDF (default: {DEFAULT_URDF})")
    ap.add_argument("--meshed-urdf", type=Path,
                    help="load this URDF with its original mesh visuals instead of spheres")
    ap.add_argument("--gui", action="store_true", help="open the interactive PyBullet GUI")
    ap.add_argument("--hold", action="store_true", help="keep the GUI open after playback")
    ap.add_argument("--png", type=Path, help="save the final frame as a PNG")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--frames-per-segment", type=int, default=4)
    ap.add_argument("--left-goal", default="0.62,0.20,1.1344")
    ap.add_argument("--right-goal", default="0.62,-0.20,1.1344")
    args = ap.parse_args()
    if args.speed <= 0 or args.fps <= 0 or args.frames_per_segment < 1:
        ap.error("speed/fps must be positive and frames-per-segment must be >= 1")

    joint_names, path = read_path(args.path)
    left_goal = np.fromstring(args.left_goal, sep=",")
    right_goal = np.fromstring(args.right_goal, sep=",")
    if left_goal.size != 3 or right_goal.size != 3:
        ap.error("goals must be x,y,z")

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to PyBullet")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(2.1, 300, -18, (0.30, 0.0, 1.25))
    p.setGravity(0, 0, -9.81)
    add_ground()
    add_shelf()
    add_marker(left_goal, (0.15, 0.95, 0.25, 0.9))
    add_marker(right_goal, (1.0, 0.72, 0.1, 0.9))

    temporary = None
    if args.meshed_urdf:
        robot_urdf = args.meshed_urdf
    else:
        temporary = tempfile.TemporaryDirectory(prefix="reachy2_cbf_viz_")
        robot_urdf = Path(temporary.name) / "reachy2_spheres.urdf"
        sphere_visual_urdf(args.urdf, robot_urdf)
    robot = p.loadURDF(str(robot_urdf), basePosition=(0, 0, GROUNDED_BASE_Z), useFixedBase=True,
                       flags=p.URDF_USE_INERTIA_FROM_FILE)

    joints = {}
    links = {}
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        joints[info[1].decode()] = j
        links[info[12].decode()] = j
    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise KeyError(f"URDF is missing path joints: {missing}")
    indices = [joints[name] for name in joint_names]
    left_tip = links["l_arm_tip_bottom"]
    right_tip = links["r_arm_tip_bottom"]

    traces = [[], []]

    def show(q):
        for joint, value in zip(indices, q):
            p.resetJointState(robot, joint, float(value))
        p.performCollisionDetection()
        tips = []
        for link in (left_tip, right_tip):
            tips.append(np.asarray(p.getLinkState(robot, link, computeForwardKinematics=True)[4]))
        for k, xyz in enumerate(tips):
            if traces[k]:
                p.addUserDebugLine(traces[k][-1], xyz,
                                   (0.1, 0.8, 1.0) if k == 0 else (1.0, 0.4, 0.1),
                                   lineWidth=3, lifeTime=0)
            traces[k].append(xyz)
        return tips

    show(path[0])
    delay = 1.0 / (args.fps * args.speed)
    for qa, qb in zip(path[:-1], path[1:]):
        for alpha in np.linspace(0, 1, args.frames_per_segment + 1)[1:]:
            show((1-alpha)*qa + alpha*qb)
            if args.gui:
                time.sleep(delay)

    final_tips = show(path[-1])
    print(f"replayed {len(path)} path states ({len(joint_names)} joints)")
    print("final left tip: ", *[f"{x:.4f}" for x in final_tips[0]],
          f" error={np.linalg.norm(final_tips[0]-left_goal):.4f} m")
    print("final right tip:", *[f"{x:.4f}" for x in final_tips[1]],
          f" error={np.linalg.norm(final_tips[1]-right_goal):.4f} m")

    if args.png:
        width, height = 1280, 900
        view = p.computeViewMatrixFromYawPitchRoll((0.30, 0.0, 1.25), 2.1, 300, -18, 0, 2)
        projection = p.computeProjectionMatrixFOV(55, width/height, 0.05, 5.0)
        image = p.getCameraImage(width, height, view, projection,
                                 renderer=p.ER_TINY_RENDERER)[2]
        from PIL import Image
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(args.png)
        print(f"saved {args.png}")

    if args.gui and args.hold:
        print("GUI open; press Ctrl-C to exit")
        try:
            while p.isConnected():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    p.disconnect()
    if temporary:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
