#!/usr/bin/env python3
"""Measure where Reachy2's right arm can put r_arm_tip, in the world frame.

Every scene coordinate in ``reachy_nav/envs.py`` is derived from this. Re-run it
after any change to the URDF, the joint limits or the base offset -- siting
scenes from the UR5's table coordinates instead is what makes goals silently
unreachable.

    python scripts/reach_envelope.py --samples 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_nav import ARM_JOINTS, ReachyArm            # noqa: E402

DEFAULT_URDF = (Path(__file__).resolve().parents[3] /
                "reachy_sbc_experiment/safe_bubble_cover/robots/reachy2/"
                "reachy2_spherized.urdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--save", type=Path, help="write the sampled tips to .npy")
    args = ap.parse_args()

    client = p.connect(p.DIRECT)
    arm = ReachyArm(args.urdf, client)
    rng = np.random.default_rng(args.seed)

    lines = ["joint limits:"]
    for name, lo, hi in zip(ARM_JOINTS, arm.lo, arm.hi):
        lines.append(f"  {name:20s} [{lo:+.4f}, {hi:+.4f}]")

    q = np.stack([rng.uniform(lo, hi, args.samples)
                  for lo, hi in zip(arm.lo, arm.hi)], axis=1)
    tips = np.empty((args.samples, 3))
    arm.zero()
    for k in range(args.samples):
        arm.set_config(q[k])
        tips[k] = arm.tip_position()

    lines.append(f"\nr_arm_tip over {args.samples} uniform in-limits samples "
                 f"(world frame, base_link at z={arm.spheres and 0.1075}):")
    for a, lab in enumerate("xyz"):
        c = tips[:, a]
        lines.append(f"  {lab}: {c.min():+.3f} .. {c.max():+.3f}   "
                     f"p5={np.percentile(c,5):+.3f} "
                     f"p50={np.percentile(c,50):+.3f} "
                     f"p95={np.percentile(c,95):+.3f}")
    lines.append("\nforward reach falls off a cliff:")
    for xc in (0.35, 0.45, 0.50, 0.55, 0.60, 0.65):
        m = tips[:, 0] > xc
        lines.append(f"  x > {xc:.2f}: {100*m.mean():5.2f}%" +
                     (f"   z {tips[m,2].min():+.3f}..{tips[m,2].max():+.3f}"
                      if m.any() else "   (unreachable)"))
    if args.save:
        np.save(args.save, tips)
        lines.append(f"\nsaved {args.save}")
    p.disconnect(client)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
