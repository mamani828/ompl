#!/usr/bin/env python3
"""Paired summary of the Baxter lever-arm bound ablation.

Reads the CSVs `run_lever_ablation.sh` writes -- `<mode>_s<seed>.csv` for the
`tight` (swept-enclosure) and `arm_length` (triangle-inequality) tables -- and
reports, per CBF method, the paired ratio on problems *both* modes solved.

Pairing is the point: these scenes have a heavy tail, so an unpaired median over
a different subset of problems per arm says more about which problems each solved
than about the bounds. Only rows where both modes returned a solution count, and
the ratio is taken per problem before any median.

    ./scripts/summarize_lever_ablation.py results/mbm_baxter_lever
"""

import statistics
import sys
from pathlib import Path

import pandas as pd

MODES = ("arm_length", "tight")
# The rows the lever-arm table can actually move. `isSafe` never touches the
# barrier and VAMP has its own collision checker, so both are invariant here and
# are reported only as a drift check on the machine.
CBF_METHODS = ("cbfRRTC", "cbfNoCert", "qpFreeGate")
INVARIANT = ("isSafe", "VAMP")
KEY = ["seed", "scene", "problem"]


def load(directory: Path, mode: str) -> pd.DataFrame:
    paths = sorted(directory.glob(f"csv/{mode}_s*.csv"))
    if not paths:
        raise SystemExit(f"no CSVs for mode {mode} under {directory}/csv")
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def paired(loose: pd.DataFrame, tight: pd.DataFrame, method: str, column: str):
    """Per-problem loose/tight ratios over problems both modes solved."""
    a = loose[(loose.method == method) & (loose.solved == 1)].set_index(KEY)
    b = tight[(tight.method == method) & (tight.solved == 1)].set_index(KEY)
    shared = a.index.intersection(b.index)
    if not len(shared):
        return [], 0
    x = a.loc[shared, column].astype(float)
    y = b.loc[shared, column].astype(float)
    keep = y > 0
    return (x[keep] / y[keep]).tolist(), int(len(shared))


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mbm_baxter_lever")
    loose, tight = (load(directory, m) for m in MODES)

    print(f"{directory}: arm_length {len(loose)} rows, tight {len(tight)} rows\n")

    print("Solve counts (eligible problems, all seeds pooled)")
    print(f"{'method':11s} {'arm_length':>12s} {'tight':>12s} "
          f"{'audit-safe a_l':>15s} {'audit-safe tight':>17s}")
    for method in CBF_METHODS + INVARIANT:
        row = []
        for frame in (loose, tight):
            sub = frame[(frame.method == method) & (frame.eligible == 1)]
            solved = sub[sub.solved == 1]
            row.append((len(solved), len(sub), len(solved[solved.unsafe_states == 0])))
        print(f"{method:11s} {row[0][0]:>7d}/{row[0][1]:<4d} {row[1][0]:>7d}/{row[1][1]:<4d} "
              f"{row[0][2]:>15d} {row[1][2]:>17d}")

    print("\nPaired arm_length/tight ratios -- above 1.0 means the tight bound won")
    print(f"{'method':11s} {'paired n':>9s} {'time':>8s} {'samples':>9s} "
          f"{'vertices':>9s} {'rad/call':>9s} {'tight faster':>13s}")
    for method in CBF_METHODS:
        cells = {}
        n = 0
        for column in ("seconds", "samples", "vertices", "rad_per_call"):
            ratios, n = paired(loose, tight, method, column)
            cells[column] = statistics.median(ratios) if ratios else float("nan")
        times, _ = paired(loose, tight, method, "seconds")
        faster = sum(1 for r in times if r > 1.0)
        share = f"{faster}/{len(times)}" if times else "-"
        # rad/call is progress per filter call, so the tight bound should *raise* it:
        # invert to keep every column reading "above 1.0 favours tight".
        radio = cells["rad_per_call"]
        print(f"{method:11s} {n:>9d} {cells['seconds']:>8.3f} {cells['samples']:>9.3f} "
              f"{cells['vertices']:>9.3f} {1.0 / radio if radio else float('nan'):>9.3f} "
              f"{share:>13s}")

    print("\nUnsafe audited states (must stay at zero for both -- a tighter bound that")
    print("returns an unsafe motion is not a tighter bound, it is a broken one)")
    for method in CBF_METHODS:
        counts = [int(f[(f.method == method) & (f.solved == 1)].unsafe_states.sum())
                  for f in (loose, tight)]
        print(f"  {method:11s} arm_length {counts[0]:>6d}   tight {counts[1]:>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
