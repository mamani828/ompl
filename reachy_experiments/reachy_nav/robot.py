"""Reachy2 right-arm kinematics + sphere collision model on PyBullet.

The mirror of ``ur5_nav/robot.py``, with three deliberate differences that are
not translations of the UR5 code but consequences of the robot:

- **No table.** Reachy is floor-standing. ``base_link`` sits at world
  z = ``BASE_Z`` (0.1075 m), the same offset ``Reachy2.h`` and
  ``visualize_reachy2_cbf.py`` use. Scene geometry is anchored to the floor.
- **The URDF's visual meshes are unusable.** They carry absolute paths from
  whoever generated the file (``/work/...`` here, ``/home/brian`` upstream), so
  every visual is replaced by the link's own collision spheres. This is also
  what the C++ side plans against, so the picture matches the model.
- **The TCP is ``r_arm_tip``**, the frame ``reachy2_symbolic_ik`` solves for --
  not ``r_hand_palm_link`` and not ``r_arm_tip_bottom``. Targeting the wrong one
  is the Reachy equivalent of the UR5 flange/TCP trap in that README, and it is
  worth the same warning: they differ by centimetres and nothing complains.
"""
from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p

BASE_Z = 0.1075
TIP_LINK = "r_arm_tip"

R_ARM = ["r_shoulder_pitch", "r_shoulder_roll", "r_elbow_yaw",
         "r_elbow_pitch", "r_wrist_roll", "r_wrist_pitch", "r_wrist_yaw"]
L_ARM = [j.replace("r_", "l_", 1) for j in R_ARM]
ARM_JOINTS = R_ARM                      # backwards compatible default

# name -> (joints, tip links). The two arms are kinematically independent, so a
# dual-arm goal is solved one arm at a time and only *checked* jointly -- which
# is where arm-vs-arm self collision actually shows up.
GROUPS = {
    "r_arm": (R_ARM, ["r_arm_tip"]),
    "l_arm": (L_ARM, ["l_arm_tip"]),
    "dual_arm": (R_ARM + L_ARM, ["r_arm_tip", "l_arm_tip"]),
}


def spheres_only_urdf(source: str | Path, dest: str | Path) -> None:
    """Rewrite a spherized URDF so every visual is its own collision sphere."""
    tree = ET.parse(str(source))
    for link in tree.getroot().iter("link"):
        for vis in list(link.findall("visual")):
            link.remove(vis)
        for col in link.findall("collision"):
            sph = col.find("geometry/sphere")
            if sph is None:
                continue
            vis = ET.SubElement(link, "visual")
            geo = ET.SubElement(vis, "geometry")
            ET.SubElement(geo, "sphere", radius=sph.get("radius"))
            org = col.find("origin")
            if org is not None:
                ET.SubElement(vis, "origin", xyz=org.get("xyz", "0 0 0"),
                              rpy=org.get("rpy", "0 0 0"))
    tree.write(str(dest))


@dataclass
class Sphere:
    link: str
    link_index: int
    offset: np.ndarray
    radius: float


class ReachyArm:
    """Right arm of Reachy2: limits, FK, sphere placement, IK."""

    def __init__(self, urdf: str | Path, client: int, group: str = "r_arm"):
        self.client = client
        self.group = group
        joint_names, tip_names = GROUPS[group]
        self._tmp = tempfile.TemporaryDirectory()
        staged = Path(self._tmp.name) / "reachy_spheres.urdf"
        spheres_only_urdf(urdf, staged)
        self.body = p.loadURDF(str(staged), basePosition=[0, 0, BASE_Z],
                               useFixedBase=True, physicsClientId=client)

        self.joint_index: Dict[str, int] = {}
        self.link_index: Dict[str, int] = {"base_link": -1}
        self.movable: List[int] = []
        self.lower: Dict[str, float] = {}
        self.upper: Dict[str, float] = {}
        for i in range(p.getNumJoints(self.body, physicsClientId=client)):
            info = p.getJointInfo(self.body, i, physicsClientId=client)
            name, jtype, link = info[1].decode(), info[2], info[12].decode()
            self.joint_index[name] = i
            self.link_index[link] = i
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable.append(i)
                self.lower[name], self.upper[name] = info[8], info[9]

        self.joint_names = list(joint_names)
        self.arm = [self.joint_index[j] for j in joint_names]
        self.tips = [self.link_index[t] for t in tip_names]
        self.tip = self.tips[0]
        self.lo = np.array([self.lower[j] for j in joint_names])
        self.hi = np.array([self.upper[j] for j in joint_names])
        # column ranges of each arm inside the group vector
        self.arm_slices = [slice(k * 7, k * 7 + 7) for k in range(len(tip_names))]
        self.spheres = self._read_spheres(urdf)

    # ---------------------------------------------------------------- spheres
    def _read_spheres(self, urdf: str | Path) -> List[Sphere]:
        out: List[Sphere] = []
        for link in ET.parse(str(urdf)).getroot().iter("link"):
            name = link.get("name")
            if name not in self.link_index:
                continue
            for col in link.findall("collision"):
                sph = col.find("geometry/sphere")
                if sph is None:
                    continue
                org = col.find("origin")
                xyz = [float(v) for v in (org.get("xyz", "0 0 0").split()
                                          if org is not None else "0 0 0".split())]
                out.append(Sphere(name, self.link_index[name],
                                  np.array(xyz), float(sph.get("radius"))))
        return out

    def set_config(self, q: Sequence[float]) -> None:
        for idx, value in zip(self.arm, q):
            p.resetJointState(self.body, idx, float(value),
                              physicsClientId=self.client)

    def zero(self) -> None:
        for idx in self.movable:
            p.resetJointState(self.body, idx, 0.0, physicsClientId=self.client)

    def tip_position(self, which: int = 0) -> np.ndarray:
        return np.asarray(p.getLinkState(self.body, self.tips[which],
                                         physicsClientId=self.client)[4])

    def sphere_centers(self) -> np.ndarray:
        """World centres of all 85 spheres at the current configuration."""
        frames: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        need = {s.link_index for s in self.spheres}
        for li in need:
            if li == -1:
                pos, orn = p.getBasePositionAndOrientation(
                    self.body, physicsClientId=self.client)
            else:
                st = p.getLinkState(self.body, li, computeForwardKinematics=1,
                                    physicsClientId=self.client)
                pos, orn = st[4], st[5]
            frames[li] = (np.asarray(pos),
                          np.asarray(p.getMatrixFromQuaternion(orn)).reshape(3, 3))
        out = np.empty((len(self.spheres), 3))
        for k, s in enumerate(self.spheres):
            pos, rot = frames[s.link_index]
            out[k] = pos + rot @ s.offset
        return out

    @property
    def radii(self) -> np.ndarray:
        return np.array([s.radius for s in self.spheres])

    # -------------------------------------------------------------------- IK
    def jacobian(self, q, h: float = 1e-5, which: int = 0,
                 cols=None) -> np.ndarray:
        """(3, 7) linear Jacobian of the tip w.r.t. the arm joints.

        Finite differences, deliberately. ``p.calculateJacobian`` was tried and
        its columns did not agree with a numerical differentiation of
        ``getLinkState``: they came back rotated -90 deg about x, so a DLS step
        built on them drove the tip sideways and IK solved 2/20 self-consistency
        cases. Seven extra FK evaluations per iteration cost far less than that
        class of bug.
        """
        q = np.asarray(q, dtype=float)
        cols = list(range(len(self.arm))) if cols is None else list(cols)
        J = np.empty((3, len(cols)))
        for k, a in enumerate(cols):
            qp = q.copy(); qp[a] += h
            qm = q.copy(); qm[a] -= h
            self.set_config(qp); pp = self.tip_position(which)
            self.set_config(qm); pm = self.tip_position(which)
            J[:, k] = (pp - pm) / (2.0 * h)
        self.set_config(q)
        return J

    def sample_pool(self, n: int = 4000, seed: int = 0, which: int = 0,
                    cols=None):
        """Precomputed (configs, tips) pool used to seed IK.

        The C++ mobile demo does the same thing for the same reason: a DLS
        solve from an arbitrary seed wanders, while one started from a sampled
        configuration whose tip is already centimetres away converges in a few
        iterations.
        """
        rng = np.random.default_rng(seed)
        cols = list(range(len(self.arm))) if cols is None else list(cols)
        q = np.zeros((n, len(self.arm)))
        for c in cols:
            q[:, c] = rng.uniform(self.lo[c], self.hi[c], n)
        tips = np.empty((n, 3))
        self.zero()
        for k in range(n):
            self.set_config(q[k])
            tips[k] = self.tip_position(which)
        return q[:, cols], tips

    def ik(self, target, seed=None, iterations: int = 200,
           tol: float = 2e-3, damping: float = 0.03, which: int = 0,
           cols=None, base=None):
        """Position-only damped-least-squares IK for ``r_arm_tip``.

        Solves over the seven arm joints only, with every other joint pinned at
        zero. Returns ``(q, error)``; ``q`` is ``None`` when the tip cannot be
        placed within ``tol``.
        """
        self.zero()
        n_all = len(self.arm)
        q = (np.asarray(base, dtype=float).copy() if base is not None
             else np.zeros(n_all))
        cols = list(range(n_all)) if cols is None else list(cols)
        if seed is not None:
            q[cols] = np.asarray(seed, dtype=float)
        q = np.clip(q, self.lo, self.hi)
        target = np.asarray(target, dtype=float)
        for _ in range(iterations):
            self.set_config(q)
            delta = target - self.tip_position(which)
            err = float(np.linalg.norm(delta))
            if err <= tol:
                return q, err
            J = self.jacobian(q, which=which, cols=cols)
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, delta)
            nn = float(np.linalg.norm(dq))
            if nn > 0.25:
                dq *= 0.25 / nn
            q[cols] = q[cols] + dq
            q = np.clip(q, self.lo, self.hi)
        self.set_config(q)
        err = float(np.linalg.norm(self.tip_position(which) - target))
        return (q if err <= tol else None), err
