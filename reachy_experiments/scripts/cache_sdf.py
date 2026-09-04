#!/usr/bin/env python3
"""Snapshot the live GP-SDF to a file so experiments share one frozen map.

    python scripts/cache_sdf.py --out out/shelf_map.npz

Sanity-checks the snapshot before writing it, because every failure mode of
this stack presents as a map that looks fine: a paused Gazebo world publishes
no depth, the mapper then ingests no scans, and every query still answers --
from whatever it last had, or from nothing at all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_nav.live_sdf import GridCache, LiveSdfEnv      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/shelf_map.npz"))
    ap.add_argument("--resolution", type=float, default=0.03)
    ap.add_argument("--mins", type=float, nargs=3, default=[-0.80, -1.00, 0.40])
    ap.add_argument("--maxs", type=float, nargs=3, default=[0.95, 1.00, 2.00])
    args = ap.parse_args()

    live = LiveSdfEnv()
    t0 = time.time()
    cache = GridCache(live, args.mins, args.maxs, args.resolution)
    fill = time.time() - t0
    obs = cache.observed_fraction()
    print(f"grid {tuple(int(d) for d in cache.dims)} = {cache.n_points} points "
          f"filled in {fill:.1f}s")
    print(f"observed (not the unobserved ceiling): {100*obs:.1f}% of cells")
    print(f"values: min {cache.values.min():+.4f}  median "
          f"{np.median(cache.values):+.4f}  max {cache.values.max():+.4f}")
    neg = float((cache.values < 0).mean())
    print(f"cells the map calls occupied: {100*neg:.2f}%")
    if neg < 1e-4:
        print("REFUSING to save: nothing is occupied anywhere in the box. "
              "The map is empty -- check the sim is unpaused and depth is "
              "flowing before trusting anything built on this.")
        return 1
    path = cache.save(args.out)
    print(f"wrote {path} ({path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
