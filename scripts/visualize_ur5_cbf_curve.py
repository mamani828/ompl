#!/usr/bin/env python3
"""Show straight-edge RRT and a curved CBF rollout on the spherized UR5.

The input is written by ``demo_UR5CBFCurve``.  No external URDF is needed: the
same 40 collision spheres used by the C++ barrier are rendered as the robot.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zlib

import numpy as np
import pybullet as p
import pybullet_data

RRT = (0.95, 0.30, 0.12, 0.75)
CBF = (0.05, 0.65, 1.00, 0.78)
OBSTACLE = (0.85, 0.10, 0.12, 0.72)
NODE = (1.00, 0.90, 0.05, 0.95)


def make_robot(radii, color):
    return [p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(
        p.GEOM_SPHERE, radius=float(radius), rgbaColor=color)) for radius in radii]


def place(robot, config):
    for body, center in zip(robot, config["centers"]):
        p.resetBasePositionAndOrientation(body, center, [0, 0, 0, 1])


def tool_trace(path):
    # The final sphere is attached to the gripper and is a useful visible proxy
    # for the end-effector trajectory.
    return np.asarray([row["centers"][-1] for row in path])


def draw_trace(points, color, width):
    for a, b in zip(points[:-1], points[1:]):
        p.addUserDebugLine(a, b, color[:3], lineWidth=width)


def draw_nodes(nodes):
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=NODE)
    for point in tool_trace(nodes):
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=point)


def draw_markers(points, color, limit=180):
    indices = np.linspace(0, len(points) - 1, min(limit, len(points))).round().astype(int)
    shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.009, rgbaColor=color)
    for point in points[indices]:
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=point)


def save_png(path):
    width, height = 1100, 800
    view = p.computeViewMatrixFromYawPitchRoll([-0.25, 0.0, 1.0], 2.6, 48, -22, 0, 2)
    projection = p.computeProjectionMatrixFOV(52, width / height, 0.05, 6.0)
    rgba = np.asarray(p.getCameraImage(width, height, view, projection,
                                      renderer=p.ER_TINY_RENDERER)[2], dtype=np.uint8)
    rgb = rgba.reshape(height, width, 4)[:, :, :3]

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    raw = b"".join(b"\0" + rgb[row].tobytes() for row in range(height))
    with open(path, "wb") as output:
        output.write(b"\x89PNG\r\n\x1a\n")
        output.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        output.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        output.write(chunk(b"IEND", b""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="JSON written by demo_UR5CBFCurve")
    parser.add_argument("--gui", action="store_true", help="open the PyBullet viewer")
    parser.add_argument("--hold", action="store_true", help="leave the final comparison visible")
    parser.add_argument("--png", metavar="PATH", help="write a headless comparison image")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    with open(args.data) as handle:
        data = json.load(handle)

    p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    for obstacle in data["obstacles"]:
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=obstacle["radius"], rgbaColor=OBSTACLE)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=obstacle["center"])

    rrt_trace = tool_trace(data["rrt_path"])
    cbf_trace = tool_trace(data["cbf_path"])
    draw_trace(rrt_trace, RRT, 4.0)
    draw_trace(cbf_trace, CBF, 5.0)
    draw_nodes(data["rrt_nodes"])
    if args.png:
        draw_markers(rrt_trace, RRT)
        draw_markers(cbf_trace, CBF)
    rrt_robot = make_robot(data["radii"], RRT)
    cbf_robot = make_robot(data["radii"], CBF)
    place(rrt_robot, data["rrt_path"][0])
    place(cbf_robot, data["cbf_path"][0])

    p.resetDebugVisualizerCamera(cameraDistance=2.6, cameraYaw=48, cameraPitch=-22,
                                 cameraTargetPosition=[-0.25, 0.0, 1.0])
    print(f"orange: RRTConnect, {len(data['rrt_nodes'])} tree-path nodes")
    print(f"blue:   CBF-RRTConnect, {len(data['cbf_nodes'])} tree-path nodes, "
          f"{len(data['cbf_path'])} executed states")
    if args.png:
        place(rrt_robot, data["rrt_path"][-1])
        place(cbf_robot, data["cbf_path"][-1])
        save_png(args.png)
        print(f"wrote {args.png}")
    if not args.gui:
        p.disconnect()
        return 0

    delay = 1.0 / max(1.0, args.fps * args.speed)
    count = max(len(data["rrt_path"]), len(data["cbf_path"]))
    for i in range(count):
        ri = round(i * (len(data["rrt_path"]) - 1) / max(1, count - 1))
        ci = round(i * (len(data["cbf_path"]) - 1) / max(1, count - 1))
        place(rrt_robot, data["rrt_path"][ri])
        place(cbf_robot, data["cbf_path"][ci])
        time.sleep(delay)
    while args.hold and p.isConnected():
        time.sleep(0.1)
    p.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
