"""RRT-Connect over the 7-DoF right arm, against the scene collision model.

This is the "a planner drops into one place" slot the UR5 harness leaves open.
It is deliberately a *baseline*: bidirectional RRT-Connect with rope
shortcutting, checking edges at the same 0.024 rad resolution the C++ demos
audit at, so its paths are directly comparable to the straight-line baseline
`motion.straight_line` measures. It is not the CBF planner -- that lives in
`demos/Reachy2CBFPlanningDemo.cpp` and consumes an SDF gradient rather than a
boolean validity check.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

RESOLUTION = 0.024          # rad, matches the C++ audit resolution


class Tree:
    """Nodes in a growing preallocated array.

    The obvious implementation keeps a Python list and does
    ``np.asarray(self.nodes) - q`` in ``nearest``. That rebuilds the whole
    array on every nearest-neighbour query, so the cost grows with the tree and
    dominates everything else: at 14 DoF it was the reason a perfectly open
    problem (78% of random samples collision-free) could not be solved in 150 s.
    """

    def __init__(self, root: np.ndarray):
        root = np.asarray(root, dtype=float)
        self._buf = np.empty((256, root.size))
        self._buf[0] = root
        self.n = 1
        self.parent: List[int] = [-1]

    @property
    def nodes(self) -> np.ndarray:
        return self._buf[:self.n]

    def nearest(self, q: np.ndarray) -> int:
        d = self._buf[:self.n] - q
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def add(self, q: np.ndarray, parent: int) -> int:
        if self.n == len(self._buf):
            self._buf = np.vstack([self._buf, np.empty_like(self._buf)])
        self._buf[self.n] = q
        self.parent.append(parent)
        self.n += 1
        return self.n - 1

    def path_to(self, i: int) -> List[np.ndarray]:
        out = []
        while i >= 0:
            out.append(self._buf[i].copy())
            i = self.parent[i]
        return out[::-1]


def edge_valid(valid: Callable[[np.ndarray], bool],
               a: np.ndarray, b: np.ndarray,
               resolution: float = RESOLUTION) -> bool:
    span = float(np.max(np.abs(b - a)))
    n = max(2, int(np.ceil(span / resolution)) + 1)
    for k in range(1, n):
        if not valid(a + (b - a) * (k / (n - 1))):
            return False
    return True


def _extend(tree: Tree, target: np.ndarray, step: float,
            valid: Callable[[np.ndarray], bool],
            resolution: float = RESOLUTION):
    i = tree.nearest(target)
    q = tree.nodes[i]
    d = target - q
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return i, True
    q_new = q + d * (min(step, n) / n)
    if not edge_valid(valid, q, q_new, resolution):
        return None, False
    return tree.add(q_new, i), bool(min(step, n) >= n - 1e-9)


def rrt_connect(start: np.ndarray, goal: np.ndarray,
                lo: np.ndarray, hi: np.ndarray,
                valid: Callable[[np.ndarray], bool],
                step: float = 0.35, timeout: float = 20.0,
                seed: int = 0,
                resolution: float = RESOLUTION) -> Optional[List[np.ndarray]]:
    """Bidirectional RRT-Connect. Returns waypoints, or None on failure."""
    if not valid(start) or not valid(goal):
        return None
    rng = np.random.default_rng(seed)
    ta, tb = Tree(np.asarray(start, float)), Tree(np.asarray(goal, float))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        q_rand = rng.uniform(lo, hi)
        ia, _ = _extend(ta, q_rand, step, valid, resolution)
        if ia is None:
            ta, tb = tb, ta
            continue
        # connect the other tree all the way to the new node
        reached = False
        while True:
            ib, done = _extend(tb, ta.nodes[ia], step, valid, resolution)
            if ib is None:
                break
            if np.linalg.norm(tb.nodes[ib] - ta.nodes[ia]) < 1e-9 or done:
                reached = True
                break
        if reached:
            pa, pb = ta.path_to(ia), tb.path_to(ib)
            path = pa + pb[::-1]
            # the trees may be swapped; orient the result to start at `start`
            if np.linalg.norm(path[0] - start) > np.linalg.norm(path[-1] - start):
                path = path[::-1]
            return path
        ta, tb = tb, ta
    return None


def shortcut(path: List[np.ndarray], valid: Callable[[np.ndarray], bool],
             iterations: int = 200, seed: int = 0,
             resolution: float = RESOLUTION) -> List[np.ndarray]:
    """Randomised shortcutting; every accepted edge is fully checked."""
    rng = np.random.default_rng(seed)
    out = [np.asarray(q, float) for q in path]
    for _ in range(iterations):
        if len(out) <= 2:
            break
        i = rng.integers(0, len(out) - 2)
        j = rng.integers(i + 2, len(out))
        if edge_valid(valid, out[i], out[j], resolution):
            out = out[:i + 1] + out[j:]
    return out


def densify(path: List[np.ndarray], resolution: float = RESOLUTION):
    """Resample a waypoint list at a fixed joint-space resolution."""
    out = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        span = float(np.max(np.abs(b - a)))
        n = max(2, int(np.ceil(span / resolution)) + 1)
        for k in range(1, n):
            out.append(a + (b - a) * (k / (n - 1)))
    return out


def length(path) -> float:
    p = [np.asarray(q, float) for q in path]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(p[:-1], p[1:])))
