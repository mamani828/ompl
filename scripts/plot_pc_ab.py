#!/usr/bin/env python3
"""CDFs for the probabilistic-completeness run: CBF rollout against ordinary planning.

Reads the per-problem CSVs `demo_UR5MBMBenchmark` writes for the pre-fix and post-fix
builds at several seeds and plots the distributions the printed min/median/max table
collapses away. Two questions, kept apart because they have different controls:

  * **versus normal** -- `cbf-rrtc` against the collision-checked baseline and VAMP, on
    the fixed build. One hue per planner, so a run missing one does not repaint the
    others.
  * **the fix itself** -- `cbf-rrtc` pre-fix against post-fix. Same planner, so the same
    hue: the build is the line style, and a reader compares like with like along a
    colour rather than across two of them.

    ./scripts/plot_pc_ab.py results/mbm_pc [outdir]

Only solved rows enter the time and path distributions; an unsolved row's `seconds` is
wherever the time limit was, not a planning cost. The CDFs are nevertheless normalised by
the number of *eligible* problems rather than solved ones, so a planner that fails some
of the set tops out below 1.0 instead of being flattered into looking complete.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour follows the planner, never its rank, and matches scripts/plot_mbm_stats.py so
# the two sets of figures can sit in one document. Slots 1-3 are the trio every run
# emits; the KPIECE pair takes slots 4-5 and is plotted on its own axes rather than
# crowded in beside them, which keeps the adjacent-pair count in any one figure at two.
PLANNER_COLOR = {
    "rrtconnect": "#2a78d6",  # blue, slot 1
    "cbf-rrtc": "#eb6834",  # orange, slot 2
    "vamp-rrtc": "#1baf7a",  # aqua, slot 3
    "kpiece": "#8b5cf6",  # violet, slot 4
    "cbf-kpiece": "#d64550",  # rose, slot 5
}

# The CSV's own names for the rows, which are not the names the table prints.
METHOD_NAME = {
    "isSafe": "rrtconnect",
    "bubbleCBF": "cbf-rrtc",
    "VAMP": "vamp-rrtc",
    "isSafeKPIECE": "kpiece",
    "bubbleCBFKPIECE": "cbf-kpiece",
}

RRT_FAMILY = ["rrtconnect", "cbf-rrtc", "vamp-rrtc"]
KPIECE_FAMILY = ["kpiece", "cbf-kpiece"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.set_ylim(0, 1.02)


def ecdf(values, denominator):
    """Empirical CDF over \\p denominator trials, so unsolved problems cost height.

    Normalising by `len(values)` instead would draw every planner reaching 1.0 and hide
    exactly the failures the figure exists to show.
    """
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0 or denominator == 0:
        return values, values
    return values, np.arange(1, len(values) + 1) / denominator


def load(root):
    """Every `<build>_s<seed>.csv` under \\p root, tagged with its build and seed."""
    frames = []
    for path in sorted(Path(root).glob("*_s[0-9].csv")):
        build, _, seed = path.stem.rpartition("_s")
        frame = pd.read_csv(path)
        frame["build"] = "post-PC" if build == "postPC" else "pre-PC"
        frame["seed"] = int(seed)
        frame["planner"] = frame["method"].map(METHOD_NAME)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no <build>_s<seed>.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def subset(data, build, planner):
    rows = data[(data["build"] == build) & (data["planner"] == planner) & (data["eligible"] == 1)]
    return rows, rows[rows["solved"] == 1]


# The CSV records seconds; every axis here is labelled in milliseconds, so the column is
# converted once, on the way in, rather than trusted to be scaled at each call site.
SCALE = {"seconds": 1e3}


def draw(ax, xs, ys, color, label, style="-"):
    ax.step(xs, ys, where="post", color=color, linewidth=2.0, linestyle=style,
            label=label, zorder=3, solid_capstyle="round")


# Heights at which successive series are direct-labelled. Anchoring every label at the
# median put them on top of each other wherever two curves crossed near it, which on a
# CDF is exactly where the interesting comparison is.
LABEL_HEIGHTS = [0.68, 0.46, 0.24, 0.86, 0.12]


def label_on_curve(ax, xs, ys, color, text, height):
    """Direct label on the curve, so identity is never colour-alone."""
    if len(xs) == 0:
        return
    index = min(int(np.searchsorted(ys, height)), len(xs) - 1)
    ax.annotate(text, (xs[index], ys[index]), textcoords="offset points",
                xytext=(7, -3), fontsize=8, color=color, zorder=4,
                fontweight="medium")


def family_figure(data, build, planners, column, xlabel, title, logx, out):
    figure, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    style_axes(ax)
    for offset, planner in enumerate(planners):
        eligible, solved = subset(data, build, planner)
        if eligible.empty:
            continue
        xs, ys = ecdf(solved[column] * SCALE.get(column, 1.0), len(eligible))
        color = PLANNER_COLOR[planner]
        draw(ax, xs, ys, color, planner)
        label_on_curve(ax, xs, ys, color, planner, LABEL_HEIGHTS[offset % len(LABEL_HEIGHTS)])
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of problems")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.tight_layout()
    figure.savefig(out, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out}")


def ab_figure(data, column, xlabel, title, logx, out):
    """The fix's own A/B: one planner, one hue, the build carried by line style."""
    figure, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    style_axes(ax)
    color = PLANNER_COLOR["cbf-rrtc"]
    for build, style in (("pre-PC", (0, (5, 2))), ("post-PC", "-")):
        eligible, solved = subset(data, build, "cbf-rrtc")
        if eligible.empty:
            continue
        xs, ys = ecdf(solved[column] * SCALE.get(column, 1.0), len(eligible))
        draw(ax, xs, ys, color, f"cbf-rrtc, {build}", style=style)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of problems")
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    legend = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.tight_layout()
    figure.savefig(out, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out}")


def summary(data, out):
    """The medians the CDFs draw, as a table, so the figures have a text counterpart."""
    records = []
    for build in sorted(data["build"].unique()):
        for planner in PLANNER_COLOR:
            eligible, solved = subset(data, build, planner)
            if eligible.empty:
                continue
            records.append(dict(
                build=build, planner=planner,
                eligible=len(eligible), solved=len(solved),
                ms_median=1e3 * solved["seconds"].median(),
                ms_mean=1e3 * solved["seconds"].mean(),
                ms_p90=1e3 * solved["seconds"].quantile(0.90),
                path_median=solved["path_length"].median(),
                vertices_median=solved["vertices"].median(),
                unsafe=int(solved["unsafe_states"].sum()),
                audited=int(solved["audited_states"].sum()),
            ))
    table = pd.DataFrame.from_records(records)
    table.to_csv(out, index=False)
    print(f"wrote {out}")
    return table


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[-4].strip())
    root = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    data = load(root)
    seeds = sorted(data["seed"].unique())
    tag = f"{len(seeds)} seeds pooled, 20 problems per scene"

    family_figure(data, "post-PC", RRT_FAMILY, "seconds",
                  "planning time, ms (log scale)", f"Planning time -- {tag}",
                  True, outdir / "cdf_time_rrt.png")
    family_figure(data, "post-PC", RRT_FAMILY, "path_length",
                  "solution path length, rad", f"Path length -- {tag}",
                  False, outdir / "cdf_path_rrt.png")
    family_figure(data, "post-PC", RRT_FAMILY, "samples",
                  "checks / filter calls / SIMD lanes (log scale)",
                  f"Sampling work -- {tag}", True, outdir / "cdf_work_rrt.png")
    family_figure(data, "post-PC", KPIECE_FAMILY, "seconds",
                  "planning time, ms (log scale)", f"Planning time, KPIECE -- {tag}",
                  True, outdir / "cdf_time_kpiece.png")

    ab_figure(data, "seconds", "planning time, ms (log scale)",
              f"Completeness fix A/B, cbf-rrtc -- {tag}", True,
              outdir / "cdf_time_pc_ab.png")
    ab_figure(data, "path_length", "solution path length, rad",
              f"Completeness fix A/B, cbf-rrtc -- {tag}", False,
              outdir / "cdf_path_pc_ab.png")

    table = summary(data, outdir / "summary.csv")
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
