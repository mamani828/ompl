#!/usr/bin/env python3
"""Render two recorded MBM motions side by side with collision spheres visible."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
UR5_ROOT = ROOT / "external/vamp/resources/ur5"
UR5_URDF = UR5_ROOT / "ur5_spherized.urdf"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def load_motion(path: Path, scene: str, index: int) -> np.ndarray:
    wanted = f"# motion {scene} {index}"
    rows: list[list[float]] = []
    active = False
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("# motion"):
                if active:
                    break
                active = line == wanted
            elif active and line and not line.startswith("#"):
                rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"{path} has no motion marked {scene} {index}")
    return np.asarray(rows, dtype=float)


def load_problem(path: Path, scene: str, index: int) -> dict:
    data = json.loads(path.read_text())
    for problem in data["problems"][scene]:
        if problem["index"] == index:
            return problem
    raise ValueError(f"{path} has no problem {scene} {index}")


class World:
    def __init__(self, problem: dict, width: int, height: int):
        self.client = p.connect(p.DIRECT)
        self.width = width
        self.height = height
        p.setAdditionalSearchPath(str(UR5_ROOT), physicsClientId=self.client)
        self.robot = p.loadURDF(
            str(UR5_URDF), useFixedBase=True, physicsClientId=self.client
        )
        name_to_joint = {}
        for joint in range(p.getNumJoints(self.robot, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot, joint, physicsClientId=self.client)
            name_to_joint[info[1].decode()] = joint
        self.joints = [name_to_joint[name] for name in JOINT_NAMES]
        self.obstacles = self._add_obstacles(problem)
        self.spheres = self._add_sphere_overlays()

    def _add_obstacles(self, problem: dict) -> list[int]:
        bodies = []
        brown = (0.52, 0.34, 0.18, 1.0)
        blue = (0.24, 0.38, 0.68, 1.0)
        for obj in problem.get("box", []):
            half = obj["half_extents"]
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.client
            )
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=brown, physicsClientId=self.client
            )
            bodies.append(
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=obj["position"],
                    baseOrientation=obj["orientation_quat_xyzw"],
                    physicsClientId=self.client,
                )
            )
        for obj in problem.get("cylinder", []):
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=obj["radius"],
                height=obj["length"],
                physicsClientId=self.client,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=obj["radius"],
                length=obj["length"],
                rgbaColor=blue,
                physicsClientId=self.client,
            )
            bodies.append(
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=obj["position"],
                    baseOrientation=obj["orientation_quat_xyzw"],
                    physicsClientId=self.client,
                )
            )
        return bodies

    def _add_sphere_overlays(self) -> list[tuple[int, int, tuple, tuple]]:
        overlays = []
        for link in range(-1, p.getNumJoints(self.robot, physicsClientId=self.client)):
            for shape in p.getCollisionShapeData(
                self.robot, link, physicsClientId=self.client
            ):
                if shape[2] != p.GEOM_SPHERE:
                    continue
                radius = float(shape[3][0])
                collision = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=radius, physicsClientId=self.client
                )
                visual = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=(0.05, 0.85, 1.0, 0.32),
                    physicsClientId=self.client,
                )
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    physicsClientId=self.client,
                )
                overlays.append((body, link, shape[5], shape[6]))
        return overlays

    def set_configuration(self, q: np.ndarray) -> int:
        for joint, value in zip(self.joints, q):
            p.resetJointState(
                self.robot, joint, float(value), physicsClientId=self.client
            )
        p.performCollisionDetection(physicsClientId=self.client)
        hits = 0
        for body, link, local_position, local_orientation in self.spheres:
            if link < 0:
                frame = p.getBasePositionAndOrientation(
                    self.robot, physicsClientId=self.client
                )
            else:
                state = p.getLinkState(
                    self.robot,
                    link,
                    computeForwardKinematics=True,
                    physicsClientId=self.client,
                )
                frame = (state[4], state[5])
            world_position, world_orientation = p.multiplyTransforms(
                frame[0], frame[1], local_position, local_orientation
            )
            p.resetBasePositionAndOrientation(
                body, world_position, world_orientation, physicsClientId=self.client
            )
            colliding = any(
                p.getClosestPoints(
                    body, obstacle, distance=0.0, physicsClientId=self.client
                )
                for obstacle in self.obstacles
            )
            hits += int(colliding)
            color = (1.0, 0.04, 0.02, 0.78) if colliding else (0.05, 0.85, 1.0, 0.32)
            p.changeVisualShape(body, -1, rgbaColor=color, physicsClientId=self.client)
        return hits

    def render(self) -> np.ndarray:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.0, 0.0, 1.25),
            distance=2.45,
            yaw=305,
            pitch=-22,
            roll=0,
            upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=58,
            aspect=self.width / self.height,
            nearVal=0.05,
            farVal=8.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            shadow=1,
            physicsClientId=self.client,
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]

    def close(self) -> None:
        p.disconnect(self.client)


def label(frame: np.ndarray, title: str, hits: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 54), fill=(10, 14, 22))
    draw.text((16, 10), title, fill=(240, 244, 250))
    status = "sphere collision" if hits else "sphere clear"
    draw.text((16, 31), status, fill=(255, 70, 55) if hits else (80, 245, 150))
    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-json", type=Path, required=True)
    parser.add_argument("--scene", default="cage")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", default="CBF-RRTC")
    parser.add_argument("--right-label", default="VAMP-RRTC")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    problem = load_problem(args.problem_json, args.scene, args.index)
    left_path = load_motion(args.left, args.scene, args.index)
    right_path = load_motion(args.right, args.scene, args.index)
    panel_width, height = 640, 480
    left = World(problem, panel_width, height)
    right = World(problem, panel_width, height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{2 * panel_width}x{height}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert encoder.stdin is not None
        for frame_index in range(args.frames):
            progress = frame_index / max(args.frames - 1, 1)
            li = round(progress * (len(left_path) - 1))
            ri = round(progress * (len(right_path) - 1))
            left_hits = left.set_configuration(left_path[li])
            right_hits = right.set_configuration(right_path[ri])
            left_image = label(left.render(), args.left_label, left_hits)
            right_image = label(right.render(), args.right_label, right_hits)
            encoder.stdin.write(np.concatenate((left_image, right_image), axis=1).tobytes())
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        code = encoder.wait()
        left.close()
        right.close()
    if code:
        raise RuntimeError(f"ffmpeg exited with status {code}")
    print(f"wrote {args.output} ({args.frames / args.fps:.1f} seconds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
