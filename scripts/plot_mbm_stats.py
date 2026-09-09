#!/usr/bin/env python3
"""Histograms and CDFs of per-problem MotionBenchMaker results.

Reads the CSV `demo_UR5MBMBenchmark` writes when given a `csvPath` argument
(one row per scene/index/planner, including skipped problems) and plots the
distributions the printed min/median/max table collapses away: planning time,
vertex count and sampling work, as a histogram and an empirical CDF, overall and
per scene.

    ./scripts/plot_mbm_stats.py results/mbm_buffer5mm/mbm_buffer5mm.csv [outdir]

Only solved rows go into the time/vertex distributions -- an unsolved row's
"seconds" is just wherever the time limit was, not a planning cost.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order, dataviz default palette. Slots 1/2/3 are the three
# planners every run emits and are the documented all-pairs-validated trio;
# hybrid-rrtc (rarely present) sits on slot 6 rather than a neighbouring hue so it
# stays separable from vamp's aqua. Colour follows the planner, never its rank, so
# a run missing one of them does not repaint the others.
PLANNER_COLOR = {
    "rrtconnect": "#2a78d6",  # blue, slot 1
    "cbf-rrtc": "#eb6834",  # orange, slot 2
    "vamp-rrtc": "#1baf7a",  # aqua, slot 3
    "hybrid-rrtc": "#008300",  # green, slot 6
}
PLANNERS = ["rrtconnect", "cbf-rrtc", "vamp-rrtc", "hybrid-rrtc"]

# One CSV column, three meanings -- collision checks for the checked baseline, filter
# calls for the CBF rollout, SIMD configuration lanes for VAMP. They measure sampling
# work; the wall-time figure is where their very different per-sample costs show up.
EVAL_LABEL = "checks / filter calls / SIMD lanes (log scale)"

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


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return values, values
    fractions = np.arange(1, len(values) + 1) / len(values)
    return values, fractions


def histogram_panel(ax, groups, log_x, xlabel, title):
    finite = [g[g > 0] for g in groups.values() if len(g[g > 0])]
    if not finite:
        ax.text(0.5, 0.5, "no solved problems", ha="center", va="center",
                fontsize=8, color=INK_SECONDARY, transform=ax.transAxes)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    combined = np.concatenate(finite)
    lo, hi = combined.min(), combined.max()
    if log_x and lo > 0:
        bins = np.logspace(np.log10(lo), np.log10(hi), 25)
    else:
        bins = np.linspace(lo, hi, 25)
    for planner, values in groups.items():
        values = values[values > 0]
        if len(values) == 0:
            continue
        ax.hist(values, bins=bins, color=PLANNER_COLOR[planner], alpha=0.55,
                label=f"{planner} (n={len(values)})", edgecolor=PLANNER_COLOR[planner],
                linewidth=0.8)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    style_axes(ax)


def cdf_panel(ax, groups, log_x, xlabel, title):
    any_data = False
    for planner, values in groups.items():
        values = values[values > 0] if log_x else values
        x, y = ecdf(values)
        if len(x) == 0:
            continue
        any_data = True
        ax.step(x, y, where="post", color=PLANNER_COLOR[planner], linewidth=2,
                label=f"{planner} (n={len(values)})")
    if not any_data:
        ax.text(0.5, 0.5, "no solved problems", ha="center", va="center",
                fontsize=8, color=INK_SECONDARY, transform=ax.transAxes)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("fraction solved ≤ x", fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    style_axes(ax)


# This branch's CSV names the columns and the methods differently from the one this
# script was first written against. Normalise on read rather than teaching every
# figure two vocabularies -- and keep the printed table's names, since those are what
# the run log and every report say.
METHOD_NAMES = {
    "isSafe": "rrtconnect",
    "bubbleCBF": "cbf-rrtc",
    "VAMP": "vamp-rrtc",
    "vamp": "vamp-rrtc",
}
COLUMN_NAMES = {"samples": "evaluations", "problem": "index", "method": "planner"}


def normalise(df):
    df = df.rename(columns={k: v for k, v in COLUMN_NAMES.items() if k in df.columns})
    if "planner" in df.columns:
        df["planner"] = df["planner"].replace(METHOD_NAMES)
    # `eligible` is the negation of the older `skipped`; one of the two is always there.
    if "skipped" not in df.columns and "eligible" in df.columns:
        df["skipped"] = 1 - df["eligible"]
    return df


def present_planners(df):
    """The fixed order, restricted to planners this run actually emitted -- so a run
    without `hybrid-rrtc` gets no empty legend entry, and the survivors keep their
    colours."""
    return [planner for planner in PLANNERS if (df["planner"] == planner).any()]


def by_planner(df, column, planners=None):
    return {planner: df.loc[df["planner"] == planner, column].to_numpy()
            for planner in (planners if planners is not None else PLANNERS)}


def make_figure(df, column, xlabel, log_x, scenes, out_path, suptitle):
    n = len(scenes) + 1
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(4.2 * cols, 3.4 * rows))
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes)

    planners = present_planners(df)
    panels = [("all scenes", df)] + [(scene, df[df["scene"] == scene]) for scene in scenes]
    for i, (label, subset) in enumerate(panels):
        r, c = divmod(i, cols)
        hist_ax = axes[r, 2 * c]
        cdf_ax = axes[r, 2 * c + 1]
        groups = by_planner(subset, column, planners)
        histogram_panel(hist_ax, groups, log_x, xlabel, label)
        cdf_panel(cdf_ax, groups, log_x, xlabel, label)
        if i == 0:
            hist_ax.legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY)

    for i in range(len(panels), rows * cols):
        r, c = divmod(i, cols)
        axes[r, 2 * c].set_visible(False)
        axes[r, 2 * c + 1].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def label_at_median(ax, groups, log_x):
    """Name each curve where it crosses the median, so the figure reads without the
    legend -- the curves converge at the top right, but their medians are exactly
    what separates them."""
    for planner, values in groups.items():
        values = values[values > 0] if log_x else values
        if len(values) == 0:
            continue
        median = float(np.median(values))
        # Haloed, because the label sits over the curve it names and a plain glyph on a
        # 2px line is unreadable at the crossing.
        ax.annotate(planner, xy=(median, 0.5), xytext=(7, 9), textcoords="offset points",
                    fontsize=8, color=INK, ha="left", va="bottom",
                    path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])
        ax.plot([median], [0.5], marker="o", markersize=5, color=PLANNER_COLOR[planner],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)


def cdf_headline_figure(df, column, xlabel, log_x, out_path, title):
    """One big CDF, all scenes pooled -- the single-plot answer to "the CDF"."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)
    groups = by_planner(df, column, present_planners(df))
    cdf_panel(ax, groups, log_x, xlabel, "")
    label_at_median(ax, groups, log_x)
    ax.set_title(title, fontsize=12, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def cdf_grid_figure(df, column, xlabel, log_x, scenes, out_path, suptitle):
    """CDF-only small multiples, one panel per scene -- full width, no histogram
    sharing the panel, for reading the curves without the min/median/max table."""
    cols = 4
    n = len(scenes) + 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    planners = present_planners(df)
    panels = [("all scenes", df)] + [(scene, df[df["scene"] == scene]) for scene in scenes]
    for i, (label, subset) in enumerate(panels):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        groups = by_planner(subset, column, planners)
        cdf_panel(ax, groups, log_x, xlabel, label)
        if i == 0:
            ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

    for i in range(len(panels), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def solve_rate_figure(df, scenes, out_path, suptitle):
    fig, ax = plt.subplots(figsize=(1.4 * max(len(scenes), 1) + 2, 4))
    fig.patch.set_facecolor(SURFACE)
    planners = present_planners(df)
    width = 0.8 / len(planners)
    x = np.arange(len(scenes))
    no_attempts = []
    for i, planner in enumerate(planners):
        rates = []
        for j, scene in enumerate(scenes):
            attempted = df[(df.scene == scene) & (df.planner == planner) & (df.skipped == 0)]
            if len(attempted):
                rates.append(100.0 * attempted.solved.mean())
            else:
                rates.append(np.nan)
                no_attempts.append(j)
        ax.bar(x + (i - (len(planners) - 1) / 2) * width, rates, width, color=PLANNER_COLOR[planner],
               label=planner, zorder=3)
    for j in set(no_attempts):
        ax.text(j, 2, "no attempts\n(all skipped)", ha="center", va="bottom", fontsize=7,
                color=INK_SECONDARY, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("solve rate (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title(suptitle, fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_quantile_table(solved, planners, scenes, out_path):
    """The numbers behind the curves: the CDF read at five points, per scene and
    planner. A figure that cannot be queried is half a result."""
    metrics = [("seconds_ms", "time_ms"), ("vertices", "vertices"),
               ("evaluations", "evaluations")]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    for scene_label, subset in [("all scenes", solved)] + [(s, solved[solved.scene == s])
                                                           for s in scenes]:
        for planner in planners:
            values = subset[subset.planner == planner]
            if not len(values):
                continue
            row = {"scene": scene_label, "planner": planner, "solved": len(values)}
            for column, name in metrics:
                for q in quantiles:
                    row[f"{name}_p{int(q * 100)}"] = float(values[column].quantile(q))
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} results.csv [outdir] [planner,planner,...]",
              file=sys.stderr)
        return 1

    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = csv_path.stem

    df = normalise(pd.read_csv(csv_path))
    if len(sys.argv) > 3:
        keep = sys.argv[3].split(",")
        df = df[df["planner"].isin(keep)]
    df["seconds_ms"] = df["seconds"] * 1e3
    solved = df[df["solved"] == 1].copy()
    scenes = sorted(df["scene"].unique())
    planners = present_planners(df)

    problems = len(df.drop_duplicates(["scene", "index"]))
    print(f"{csv_path}: {problems} problems x {len(planners)} planners "
          f"({', '.join(planners)}), {len(solved)} solved rows, "
          f"{df.loc[df.planner == planners[0], 'skipped'].sum()} skipped, {len(scenes)} scenes")

    make_figure(solved, "seconds_ms", "planning time (ms, log scale)", True, scenes,
                out_dir / f"{run_name}_time.png", f"{run_name} -- planning time, solved problems only")
    make_figure(solved, "vertices", "tree vertices (log scale)", True, scenes,
                out_dir / f"{run_name}_vertices.png",
                f"{run_name} -- vertex count, solved problems only")
    make_figure(solved, "evaluations", EVAL_LABEL, True, scenes,
                out_dir / f"{run_name}_evaluations.png",
                f"{run_name} -- sampling work, solved problems only")
    solve_rate_figure(df, scenes, out_dir / f"{run_name}_solve_rate.png",
                      f"{run_name} -- solve rate by scene (attempted, not skipped)")

    cdf_headline_figure(solved, "seconds_ms", "planning time (ms, log scale)", True,
                        out_dir / f"{run_name}_time_cdf.png",
                        f"{run_name} -- planning time CDF, all scenes pooled")
    cdf_headline_figure(solved, "vertices", "tree vertices (log scale)", True,
                        out_dir / f"{run_name}_vertices_cdf.png",
                        f"{run_name} -- vertex count CDF, all scenes pooled")
    cdf_grid_figure(solved, "seconds_ms", "planning time (ms, log scale)", True, scenes,
                    out_dir / f"{run_name}_time_cdf_by_scene.png",
                    f"{run_name} -- planning time CDF by scene")
    cdf_grid_figure(solved, "vertices", "tree vertices (log scale)", True, scenes,
                    out_dir / f"{run_name}_vertices_cdf_by_scene.png",
                    f"{run_name} -- vertex count CDF by scene")
    cdf_headline_figure(solved, "evaluations", EVAL_LABEL, True,
                        out_dir / f"{run_name}_evaluations_cdf.png",
                        f"{run_name} -- sampling work CDF, all scenes pooled")
    cdf_grid_figure(solved, "evaluations", EVAL_LABEL, True, scenes,
                    out_dir / f"{run_name}_evaluations_cdf_by_scene.png",
                    f"{run_name} -- sampling work CDF by scene")

    write_quantile_table(solved, planners, scenes, out_dir / f"{run_name}_quantiles.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
