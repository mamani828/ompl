"""Analytic obstacle primitives and their exact signed distance field.

The field is evaluated from the primitives, never sampled from collision
queries, for the reason ``ur5_experiments/scripts/export_scene.py`` gives: the
CBF consumes the SDF *gradient*, and only an analytic field gets that right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class Box:
    """Axis-aligned box: centre and half-extents, both in world metres."""
    center: Sequence[float]
    half: Sequence[float]
    rgba: Sequence[float] = (0.55, 0.35, 0.18, 1.0)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        q = np.abs(pts - np.asarray(self.center)) - np.asarray(self.half)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside


@dataclass
class Goal:
    """A Cartesian tip target plus the direction the arm should come from."""
    position: Sequence[float]
    approach: str = "+x"
    label: str = ""


@dataclass
class Env:
    name: str
    description: str
    boxes: List[Box] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance to the nearest obstacle; negative inside one."""
        pts = np.atleast_2d(pts)
        if not self.boxes:
            return np.full(len(pts), np.inf)
        return np.min(np.stack([b.distance(pts) for b in self.boxes]), axis=0)

    def clearance(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Per-sphere clearance: obstacle distance minus the sphere radius."""
        return self.distance(centers) - radii

    def spawn(self, client: int) -> List[int]:
        import pybullet as p
        ids = []
        for b in self.boxes:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(b.half),
                                         physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(b.half),
                                      rgbaColor=list(b.rgba), physicsClientId=client)
            ids.append(p.createMultiBody(0, col, vis, list(b.center),
                                         physicsClientId=client))
        return ids
