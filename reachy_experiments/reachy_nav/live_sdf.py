"""The live GP-SDF from `sdf_mapping` as a drop-in replacement for `Env`.

`Env` answers `distance(points)` from analytic primitives. This answers the
same question from the map `sdf_mapping` builds out of the head ToF stream, so
the planner in `planner.py` can be run against **sensed** geometry without
changing a line of it.

Two things it has to get right, and both fail silently when wrong:

- **Frames.** The map is in `odom`. This harness places `base_link` at world
  z = `BASE_Z` (0.1075) while Gazebo places it at odom z = 0, so a harness
  point converts to odom by subtracting that. Verified against TF rather than
  assumed: `torso` reads (-0.010, 0, 1.1035) here and (-0.010, 0, 0.996) in
  Gazebo, a difference of exactly 0.1075.
- **Unobserved space.** The map reports a large positive distance where it has
  seen nothing, so free space and never-looked-at space are indistinguishable.
  `unknown_distance` caps what an unobserved reading may contribute; the
  default keeps it finite and generous, which is optimistic, and is the reason
  a plan against this map is not the same promise as a plan against `Env`.

The self-collision channel is untouched: the robot is self-filtered out of the
depth stream on purpose, so it does not appear in the map and the analytic
sphere-pair model remains the only thing checking the robot against itself.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .robot import BASE_Z

# The stack's own client, reused rather than re-implemented. pycapnp 2.x runs
# the KJ loop on asyncio and neither the client nor its capabilities may cross
# threads, so it owns one thread running one loop inside one kj_loop context
# for the client's lifetime. Getting that wrong is a "no running event loop"
# at best and a std::terminate at worst.
# Located the same way every script here locates the spherized URDF: this repo
# and reachy_sbc_experiment are siblings. $REACHY_SBC_ROOT overrides it, for a
# checkout that does not follow that layout.
SBC = Path(os.environ.get(
    "REACHY_SBC_ROOT",
    Path(__file__).resolve().parents[3] / "reachy_sbc_experiment"))
if str(SBC) not in sys.path:
    sys.path.insert(0, str(SBC))

SCHEMA = SBC / "config/capnp/sdf_query.capnp"
HOST, PORT = "127.0.0.1", 51111
# world -> odom. Measured, not assumed; see the module docstring.
WORLD_TO_ODOM = np.array([0.0, 0.0, -BASE_Z])


class LiveSdfEnv:
    """Duck-typed `Env`: `.distance(pts)` and `.clearance(centers, radii)`."""

    name = "live_sdf"
    description = "GP-SDF built from the head ToF stream in Gazebo"
    boxes: list = []

    def __init__(self, host: str = HOST, port: int = PORT,
                 schema: Path = SCHEMA, unknown_distance: float = 2.0,
                 batch: int = 4000, connect_timeout: float = 30.0):
        from sbc_planner.capnp_sdf import CapnpSdfClient
        self._client = CapnpSdfClient(host=host, port=port,
                                      schema_path=str(schema),
                                      connect_timeout_sec=connect_timeout)
        self._unknown = float(unknown_distance)
        self._batch = int(batch)
        self._lock = threading.Lock()
        self.queries = 0
        self.points = 0

    def _query(self, pts_odom: np.ndarray) -> np.ndarray:
        return np.asarray(self._client.query(pts_odom), dtype=float)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance to the sensed surface, for world-frame points."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        odom = pts + WORLD_TO_ODOM
        out = np.empty(len(odom))
        with self._lock:
            for s in range(0, len(odom), self._batch):
                chunk = odom[s:s + self._batch]
                out[s:s + len(chunk)] = self._query(chunk)
            self.queries += 1
            self.points += len(odom)
        # Unobserved space comes back as a large positive number; cap it so it
        # cannot certify more clearance than anything was ever measured to have.
        return np.minimum(out, self._unknown)

    def clearance(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        return self.distance(centers) - np.asarray(radii, dtype=float)

    def stats(self) -> str:
        return (f"{self.queries} queries, {self.points} points "
                f"({self.points / max(self.queries,1):.1f} per query)")


class GridCache:
    """Prefetch the live map onto a grid and interpolate it in numpy.

    A single RPC costs ~8 ms regardless of size, so a per-configuration query
    makes planning ~12x slower than the analytic check. The production planner
    solves this the same way (`sbc_planner/cached_sdf.py`): sample once, then
    answer from the grid.

    Trilinear, with no attempt at the conservative variant. On this map the
    discretisation error is small next to the map's own errors, which are the
    thing the experiment is measuring.
    """

    name = "live_sdf_cached"
    description = "GP-SDF from the head ToF, sampled onto a grid"
    boxes: list = []

    def __init__(self, env: "LiveSdfEnv", mins, maxs, resolution: float = 0.04):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        self.dims = np.maximum(
            2, np.ceil((self.maxs - self.mins) / resolution).astype(int) + 1)
        self.step = (self.maxs - self.mins) / (self.dims - 1)
        axes = [np.linspace(self.mins[i], self.maxs[i], self.dims[i])
                for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        self.values = env.distance(pts).reshape(self.dims)
        self.n_points = len(pts)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        g = (pts - self.mins) / self.step
        g = np.clip(g, 0.0, np.array(self.dims, dtype=float) - 1.000001)
        i0 = np.floor(g).astype(int)
        f = g - i0
        i1 = np.minimum(i0 + 1, np.array(self.dims) - 1)
        v = self.values
        out = np.zeros(len(pts))
        for dx in (0, 1):
            ix = i0[:, 0] if dx == 0 else i1[:, 0]
            wx = (1 - f[:, 0]) if dx == 0 else f[:, 0]
            for dy in (0, 1):
                iy = i0[:, 1] if dy == 0 else i1[:, 1]
                wy = (1 - f[:, 1]) if dy == 0 else f[:, 1]
                for dz in (0, 1):
                    iz = i0[:, 2] if dz == 0 else i1[:, 2]
                    wz = (1 - f[:, 2]) if dz == 0 else f[:, 2]
                    out += wx * wy * wz * v[ix, iy, iz]
        return out

    def clearance(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        return self.distance(centers) - np.asarray(radii, dtype=float)

    # ---------------------------------------------------------------- disk
    def save(self, path) -> Path:
        """Freeze this snapshot so experiments share one identical map.

        Worth doing even though a refill takes under a second: the map moves.
        It drifts while it is filling, it is not reproducible across mapper
        restarts (the same static scene gave +0.0505 .. +0.0723 m at one probe
        point in this stack's own notes), and it changes as the ToF keeps
        observing. An A/B run against two different maps measures the map, not
        the change under test.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, values=self.values, mins=self.mins,
                            maxs=self.maxs, dims=np.asarray(self.dims),
                            step=self.step)
        return path

    @classmethod
    def load(cls, path) -> "GridCache":
        d = np.load(str(path))
        obj = cls.__new__(cls)
        obj.values = d["values"]
        obj.mins, obj.maxs = d["mins"], d["maxs"]
        obj.dims, obj.step = d["dims"], d["step"]
        obj.n_points = int(np.prod(obj.dims))
        return obj

    def observed_fraction(self, unknown: float = 1.99) -> float:
        """Share of grid cells that are not just the unobserved ceiling."""
        return float((self.values < unknown).mean())
