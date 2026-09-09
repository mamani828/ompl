#!/usr/bin/env python3
"""Head-to-head CDFs for the certified-region hop rule against the no-op certificate.

Reads the per-problem CSVs `demo_UR5MBMBenchmark` writes for the two hop rules at
several decay rates -- `FilteredStateSpace::setSafeHops()` off and on -- and plots
the distributions the min/median/max table collapses away, for the `cbf-rrtc` row
only. The other two planner rows are untouched by the flag and serve as the control.

    ./scripts/plot_region_ab.py results/mbm_region [outdir]

Writes one CDF pair per metric (planning time, barrier evaluations) with all decay
rates overlaid, plus a summary CSV of the medians.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# One hue per decay rate, dataviz categorical order; the hop rule is the line style, so
# a reader compares like with like along a colour and reads the effect across styles.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#d64550", "#008300"]


def discover(root):
    """Which decay rates this run actually produced, in numeric order.

    A run may sweep kappa or fix it; hard-coding the sweep meant a single-kappa run
    plotted nothing. Both files of a pair must exist, since the point is the A/B.
    """
    found = []
    for path in sorted(Path(root).glob("k*_safe.csv")):
        kappa = path.name[1:-len("_safe.csv")]
        if (Path(root) / f"k{kappa}_noop.csv").exists():
            found.append(int(kappa))
    return sorted(found)
STYLE = {"noop": (0, (4, 2)), "safe": "solid"}
LABEL = {"noop": "no-op certificate", "safe": "safety certificate (region)"}


def cdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


def load(root, kappa, mode):
    frame = pd.read_csv(Path(root) / f"k{kappa}_{mode}.csv")
    # Only solved rows: an unsolved row's "seconds" is the time limit, not a cost.
    return frame[(frame["method"] == "bubbleCBF") & (frame["solved"] == 1)]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/mbm_region"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(root) / "plots" / "ab")
    out.mkdir(parents=True, exist_ok=True)

    kappas = discover(root)
    if not kappas:
        sys.exit(f"no k*_noop.csv / k*_safe.csv pairs under {root}")
    color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(kappas)}

    metrics = [("seconds", "planning time (ms)", 1e3, True),
               ("samples", "barrier evaluations", 1.0, True)]
    rows = []

    for column, label, scale, logx in metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for kappa in kappas:
            for mode in ("noop", "safe"):
                frame = load(root, kappa, mode)
                if frame.empty:
                    continue
                x, y = cdf(frame[column] * scale)
                ax.plot(x, y, color=color[kappa], linestyle=STYLE[mode], linewidth=1.7,
                        label=f"$\\kappa$={kappa}, {LABEL[mode]}")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("fraction of problems")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
        ax.set_title(f"cbf-rrtc: {label}, 140 MotionBenchMaker problems")
        fig.tight_layout()
        fig.savefig(out / f"ab_{column}_cdf.png", dpi=170)
        plt.close(fig)

    for kappa in kappas:
        entry = {"kappa": kappa}
        for mode in ("noop", "safe"):
            frame = load(root, kappa, mode)
            allrows = pd.read_csv(Path(root) / f"k{kappa}_{mode}.csv")
            cbf = allrows[allrows["method"] == "bubbleCBF"]
            entry[f"{mode}_solved"] = int(cbf["solved"].sum())
            entry[f"{mode}_problems"] = int(len(cbf))
            entry[f"{mode}_median_ms"] = float(frame["seconds"].median() * 1e3)
            entry[f"{mode}_mean_ms"] = float(frame["seconds"].mean() * 1e3)
            entry[f"{mode}_median_evals"] = float(frame["samples"].median())
            entry[f"{mode}_mean_evals"] = float(frame["samples"].mean())
            entry[f"{mode}_median_path"] = float(frame["path_length"].median())
            entry[f"{mode}_unsafe"] = int(cbf["unsafe_states"].sum())
            entry[f"{mode}_audited"] = int(cbf["audited_states"].sum())
            entry[f"{mode}_misses"] = int(cbf["misses"].sum())
            entry[f"{mode}_self_colliding"] = int(cbf["self_colliding"].sum())
        for key in ("median_ms", "mean_ms", "median_evals", "mean_evals", "median_path"):
            a, b = entry[f"noop_{key}"], entry[f"safe_{key}"]
            entry[f"delta_{key}_pct"] = 100.0 * (b - a) / a if a else float("nan")
        rows.append(entry)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "region_ab_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out}/ab_seconds_cdf.png, {out}/ab_samples_cdf.png, "
          f"{out}/region_ab_summary.csv")


if __name__ == "__main__":
    main()
