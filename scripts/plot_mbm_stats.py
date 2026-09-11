#!/usr/bin/env python3
"""Clean CDFs of per-problem MotionBenchMaker results.

Reads the CSV `demo_UR5MBMBenchmark` writes when given a `csvPath` argument
(one row per scene/index/planner, including skipped problems) and plots the
distributions the printed min/median/max table collapses away: planning time,
vertex count and sampling work, as empirical CDFs overall and per scene.

    ./scripts/plot_mbm_stats.py results/mbm_buffer5mm/mbm_buffer5mm.csv [outdir]

Only solved rows go into the time/vertex distributions -- an unsolved row's
"seconds" is just wherever the time limit was, not a planning cost.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical order. Colour follows the planner, never its rank, so a run
# missing an optional row does not repaint the others.
PLANNER_COLOR = {
    "rrtconnect": "#2a78d6",
    "qp-fixed": "#8c62aa",
    "qp-plain": "#6f6f6f",
    "qp-lipsch": "#eb6834",
    "qp-envelope": "#13a89e",
    "qp-free": "#d43f70",
    "vamp-rrtc": "#1baf7a",
    "hybrid-rrtc": "#008300",
    # Baxter's one filtered rollout plays qp-fixed's role, so it keeps that colour;
    # no run emits both, so the two never share a legend.
    "cbf-rrtc": "#8c62aa",
    # The same rollout with the hop certificate off -- a muted version of its colour,
    # since the pair is meant to be read against each other.
    "cbf-nocert": "#b9a3cc",
}
PLANNERS = ["rrtconnect", "qp-plain", "qp-fixed", "qp-lipsch", "qp-free", "qp-envelope",
            "cbf-rrtc", "cbf-nocert", "vamp-rrtc", "hybrid-rrtc"]

DISPLAY_NAMES = {
    "rrtconnect": "RRT-Connect",
    "qp-plain": "CBF-RRT",
    "qp-fixed": "Adaptive CBF-RRT (Fixed Step)",
    "qp-lipsch": "Adaptive CBF-RRT (Lipschitz)",
    "qp-free": "CBF-RRT (No QP)",
    "qp-envelope": "Adaptive CBF-RRT (Envelope)",
    "cbf-rrtc": "CBF-RRT-Connect",
    "cbf-nocert": "CBF-RRT-Connect (No Certificate)",
    "vamp-rrtc": "VAMP-RRT-Connect",
    "hybrid-rrtc": "Hybrid RRT-Connect",
}

SCENE_NAMES = {
    "bookshelf_small": "Small Bookshelf",
    "bookshelf_tall": "Tall Bookshelf",
    "bookshelf_thin": "Narrow Bookshelf",
    "box": "Box",
    "cage": "Cage",
    "table_pick": "Tabletop Pick",
    "table_under_pick": "Under-Table Pick",
}

# One CSV column, three meanings -- collision checks for the checked baseline, filter
# calls for each CBF rollout, and SIMD configuration lanes for VAMP. They measure
# sampling work; wall time captures their very different per-evaluation costs.
EVAL_LABEL = "Evaluations"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return values, values
    fractions = np.arange(1, len(values) + 1) / len(values)
    return values, fractions


def cdf_panel(ax, groups, log_x, title):
    any_data = False
    for planner, values in groups.items():
        values = values[values > 0] if log_x else values
        x, y = ecdf(values)
        if len(x) == 0:
            continue
        any_data = True
        ax.step(x, y, where="post", color=PLANNER_COLOR[planner], linewidth=1.8,
                label=DISPLAY_NAMES.get(planner, planner))
    if not any_data:
        ax.text(0.5, 0.5, "no solved problems", ha="center", va="center",
                fontsize=8, color=INK_SECONDARY, transform=ax.transAxes)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=7)
    style_axes(ax)


# This branch's CSV names the columns and the methods differently from the one this
# script was first written against. Normalise on read rather than teaching every
# figure two vocabularies -- and keep the printed table's names, since those are what
# the run log and every report say.
METHOD_NAMES = {
    "isSafe": "rrtconnect",
    "qpFixed": "qp-fixed",
    "qpPlain": "qp-plain",
    "qpAdaptive": "qp-lipsch",
    "qpEnvelope": "qp-envelope",
    "bubbleCBF": "qp-lipsch",
    "qpFreeGate": "qp-free",
    "VAMP": "vamp-rrtc",
    "vamp": "vamp-rrtc",
    # The Baxter harness runs a single filtered rollout rather than the UR5's three
    # barrier variants, and labels it `cbfRRTC`; it prints as `cbf-rrtc`.
    "cbfRRTC": "cbf-rrtc",
    "cbfNoCert": "cbf-nocert",
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


def cdf_headline_figure(df, column, xlabel, log_x, out_path):
    """One pooled CDF with restrained annotation."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor("white")
    groups = by_planner(df, column, present_planners(df))
    cdf_panel(ax, groups, log_x, "")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY,
              loc="lower right", handlelength=2.4)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def cdf_grid_figure(df, column, xlabel, log_x, scenes, out_path):
    """CDF small multiples with shared labels and a single legend."""
    cols = 3
    n = len(scenes) + 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.7 * rows),
                             squeeze=False, sharey=True)
    fig.patch.set_facecolor("white")
    planners = present_planners(df)
    panels = [("All Scenes", df)] + [
        (SCENE_NAMES.get(scene, scene.replace("_", " ").title()), df[df["scene"] == scene])
        for scene in scenes
    ]
    for i, (label, subset) in enumerate(panels):
        r, c = divmod(i, cols)
        cdf_panel(axes[r, c], by_planner(subset, column, planners), log_x, label)
    for i in range(len(panels), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5),
               frameon=False, fontsize=9, handlelength=2.3)
    fig.supxlabel(xlabel, fontsize=10)
    fig.supylabel("Cumulative fraction", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.88,
                        wspace=0.16, hspace=0.34)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def write_quantile_table(solved, planners, scenes, out_path):
    """The numbers behind the curves: the CDF read at five points, per scene and
    planner. A figure that cannot be queried is half a result."""
    metrics = [("seconds_ms", "time_ms"), ("vertices", "vertices"),
               ("evaluations", "evaluations")]
    if "qp_rows_per_call" in solved.columns:
        metrics.append(("qp_rows_per_call", "qp_rows_per_call"))
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
    # The QP's constraint-row count, per filter call. `qp_rows` is summed over calls by
    # the benchmark, so the program size is the ratio; rows that build no QP (`rrtconnect`,
    # `qp-free`) have no calls and are left out rather than entered as zero. Absent from
    # CSVs written before the counters existed, hence the guard.
    if {"qp_rows", "qp_calls"}.issubset(df.columns):
        df["qp_rows_per_call"] = np.where(df["qp_calls"] > 0,
                                          df["qp_rows"] / df["qp_calls"].replace(0, np.nan),
                                          np.nan)
    solved = df[df["solved"] == 1].copy()
    scenes = sorted(df["scene"].unique())
    planners = present_planners(df)

    problems = len(df.drop_duplicates(["scene", "index"]))
    print(f"{csv_path}: {problems} problems x {len(planners)} planners "
          f"({', '.join(planners)}), {len(solved)} solved rows, "
          f"{df.loc[df.planner == planners[0], 'skipped'].sum()} skipped, {len(scenes)} scenes")

    cdf_headline_figure(solved, "seconds_ms", "Planning time (ms)", True,
                        out_dir / f"{run_name}_time_cdf.png")
    cdf_headline_figure(solved, "vertices", "Tree vertices", True,
                        out_dir / f"{run_name}_vertices_cdf.png")
    cdf_grid_figure(solved, "seconds_ms", "Planning time (ms)", True, scenes,
                    out_dir / f"{run_name}_time_cdf_by_scene.png")
    cdf_grid_figure(solved, "vertices", "Tree vertices", True, scenes,
                    out_dir / f"{run_name}_vertices_cdf_by_scene.png")
    cdf_headline_figure(solved, "evaluations", EVAL_LABEL, True,
                        out_dir / f"{run_name}_evaluations_cdf.png")
    cdf_grid_figure(solved, "evaluations", EVAL_LABEL, True, scenes,
                    out_dir / f"{run_name}_evaluations_cdf_by_scene.png")

    write_quantile_table(solved, planners, scenes, out_dir / f"{run_name}_quantiles.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
