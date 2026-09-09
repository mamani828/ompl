#!/usr/bin/env python3
"""Head-to-head CDFs for the Lipschitz duration certificate against a fixed step.

The certificate (`ClearanceBarrier::durations()`) is a Lipschitz bound over the
whole step, so a rollout may integrate one filtered control across it instead of
re-filtering every `stepSize`. `FilteredStateSpace::setMaxStepScale(1)` caps the hop
at the step it was asked about, which never spends the certificate and so recovers
fixed-step rollout -- the two rows differ in that cap and nothing else.

    ./scripts/plot_cert_ab.py results/mbm_cert [outdir]

Expects `csv/cert_s*.csv` (uncapped) and `csv/fixed_s*.csv` (`maxStepScale 1`) from
`demo_UR5MBMBenchmark`, pooled over seeds. Plots the `cbf-rrtc` row only; the
baseline and VAMP rows are untouched by the cap and serve as the control.

Writes, into `outdir`:

  cert_ab_cdf.png                    every metric, pooled over scenes
  cert_ab_<metric>_cdf.png           one pooled panel per metric, standalone
  cert_ab_per_scene_<metric>_cdf.png that metric faceted by scene
  cert_ab_summary.csv                pooled medians, means and safety columns
  cert_ab_per_scene.csv              the same per scene, paired problem by problem
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Two hues in the repo's categorical order, validated for CVD separation against the
# light surface. The cap is also the line style, so identity never rests on colour.
COLOR = {"cert": "#2a78d6", "fixed": "#eb6834"}
STYLE = {"cert": "solid", "fixed": (0, (4, 2))}
LABEL = {"cert": "certificate spent (uncapped)", "fixed": "fixed step (maxStepScale 1)"}
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6b68"

# (column, axis label, scale, log x). Order is the reading order of the pooled figure:
# the three cost measures first, then what the certificate does to the motion.
METRICS = [
    ("seconds", "planning time (ms)", 1e3, True),
    ("samples", "barrier evaluations", 1.0, True),
    ("vertices", "tree vertices", 1.0, True),
    ("rad_per_call", "radians of progress per filter call", 1.0, False),
    ("path_length", "solution path length (rad)", 1.0, False),
]

# Which metrics get a per-scene facet grid. The cost measures are the ones whose
# scene-to-scene spread is the point; the other two are in the per-scene CSV.
PER_SCENE = ("seconds", "samples", "vertices")

BY_COLUMN = {column: (label, scale, logx) for column, label, scale, logx in METRICS}


def cdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1) / x.size


def load(root, mode):
    """Every seed's `cbf-rrtc` rows, pooled."""
    files = sorted((Path(root) / "csv").glob(f"{mode}_s*.csv"))
    if not files:
        sys.exit(f"no {mode}_s*.csv under {root}/csv")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame["method"] == "bubbleCBF"]


def solved(frame):
    """Eligible and solved only: an ineligible problem was never attempted, and an
    unsolved row's `seconds` is the time limit rather than a cost."""
    return frame[(frame["eligible"] == 1) & (frame["solved"] == 1)]


def draw(ax, data, column, axis_labels=True):
    label, scale, logx = BY_COLUMN[column]
    for mode in ("cert", "fixed"):
        values = solved(data[mode])[column] * scale
        if values.empty:
            continue
        x, y = cdf(values)
        ax.plot(x, y, color=COLOR[mode], linestyle=STYLE[mode], linewidth=2.0,
                solid_capstyle="round", label=LABEL[mode])
        # The median, marked rather than left to be read off the axis: it is the one
        # number the table reports, so the curve should say where it sits. Drawn with a
        # surface ring, since in a scene where the two agree the marks overlap.
        median = float(np.median(values))
        ax.plot([median], [0.5], marker="o", markersize=7, color=COLOR[mode],
                markeredgecolor=SURFACE, markeredgewidth=1.6,
                zorder=5 if mode == "cert" else 4)
    if logx:
        ax.set_xscale("log")
    if axis_labels:
        ax.set_xlabel(label, color=INK, fontsize=9)
        ax.set_ylabel("fraction of problems", color=INK, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6, color=MUTED)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)


def legend(fig, ax, y):
    """Figure level rather than in a panel: every panel's lower right has curve in it,
    and a legend that covers data is worse than one the eye has to travel to."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncols=2, loc="upper center",
               bbox_to_anchor=(0.5, y), frameon=False, labelcolor=INK)


def pooled(data, out, problems):
    columns = 3
    rows = (len(METRICS) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 3.4 * rows),
                             facecolor=SURFACE)
    flat = axes.ravel()
    for ax, (column, *_) in zip(flat, METRICS):
        ax.set_facecolor(SURFACE)
        draw(ax, data, column)
    for ax in flat[len(METRICS):]:
        ax.set_visible(False)
    legend(fig, flat[0], 0.945)
    fig.suptitle("Lipschitz duration certificate vs fixed step, cbf-rrtc over "
                 f"{problems} solved MotionBenchMaker UR5 runs",
                 color=INK, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out / "cert_ab_cdf.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)

    for column, label, *_ in METRICS:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        draw(ax, data, column)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95,
                  edgecolor=MUTED, labelcolor=INK)
        ax.set_title(f"cbf-rrtc: {label}", color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"cert_ab_{column}_cdf.png", dpi=170, facecolor=SURFACE)
        plt.close(fig)


def facet(data, out, column, scenes):
    """One metric, one panel per scene, shared axes.

    Pooling hides the thing worth knowing: how far the certificate can hop is a
    property of how open the scene is, so the seven scenes are seven different answers
    and the pooled median is a mixture nobody planned in. Axes are shared so the
    panels can be read against each other rather than only within themselves.
    """
    label = BY_COLUMN[column][0]
    columns = 4
    rows = (len(scenes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.9 * rows),
                             facecolor=SURFACE, sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, scene in zip(flat, scenes):
        ax.set_facecolor(SURFACE)
        scoped = {mode: frame[frame["scene"] == scene] for mode, frame in data.items()}
        draw(ax, scoped, column, axis_labels=False)
        ax.set_title(scene, color=INK, fontsize=9)
    for ax in flat[len(scenes):]:
        ax.set_visible(False)
    # One axis label per edge rather than per panel: shared scales, so repeating them
    # seven times is ink that says nothing.
    fig.supxlabel(label, color=INK, fontsize=9)
    fig.supylabel("fraction of problems", color=INK, fontsize=9)
    legend(fig, flat[0], 0.955)
    fig.suptitle(f"{label[0].upper() + label[1:]} by scene, "
                 "certificate against fixed step", color=INK, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.92))
    fig.savefig(out / f"cert_ab_per_scene_{column}_cdf.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def pooled_table(data, out):
    entry = {}
    for mode in ("cert", "fixed"):
        rows, ok = data[mode], solved(data[mode])
        entry[f"{mode}_solved"] = int(rows["solved"].sum())
        entry[f"{mode}_eligible"] = int(rows["eligible"].sum())
        for column, _, scale, _ in METRICS:
            entry[f"{mode}_median_{column}"] = float(ok[column].median() * scale)
            entry[f"{mode}_mean_{column}"] = float(ok[column].mean() * scale)
        entry[f"{mode}_p95_seconds"] = float(ok["seconds"].quantile(0.95) * 1e3)
        entry[f"{mode}_coarse"] = float(ok["coarse_fraction"].mean())
        entry[f"{mode}_unsafe"] = int(rows["unsafe_states"].sum())
        entry[f"{mode}_audited"] = int(rows["audited_states"].sum())
        entry[f"{mode}_worst_clearance"] = float(rows["min_clearance"].min())
        entry[f"{mode}_misses"] = int(rows["misses"].sum())
        entry[f"{mode}_self_colliding"] = int(rows["self_colliding"].sum())

    key = ["seed", "scene", "problem"]
    both = solved(data["cert"]).merge(solved(data["fixed"]), on=key, suffixes=("_c", "_f"))
    entry["paired_n"] = int(len(both))
    for column, *_ in METRICS:
        ratio = both[f"{column}_f"] / both[f"{column}_c"].replace(0, np.nan)
        entry[f"paired_median_ratio_{column}"] = float(ratio.median())
    entry["paired_slower_without"] = int((both["seconds_f"] > both["seconds_c"]).sum())

    pd.DataFrame([entry]).to_csv(out / "cert_ab_summary.csv", index=False)

    print(f"pooled over {len(both)} paired runs\n")
    print(f"{'metric':<24}{'certificate':>13}{'fixed step':>13}{'fixed/cert':>12}"
          f"{'paired':>9}")
    for column, label, scale, _ in METRICS:
        c, f = entry[f"cert_median_{column}"], entry[f"fixed_median_{column}"]
        print(f"{'median ' + column:<24}{c:>13.4f}{f:>13.4f}{f / c:>12.3f}"
              f"{entry['paired_median_ratio_' + column]:>9.3f}")
    for column, label, scale, _ in METRICS:
        c, f = entry[f"cert_mean_{column}"], entry[f"fixed_mean_{column}"]
        print(f"{'mean   ' + column:<24}{c:>13.4f}{f:>13.4f}{f / c:>12.3f}")
    print(f"{'p95 seconds (ms)':<24}{entry['cert_p95_seconds']:>13.4f}"
          f"{entry['fixed_p95_seconds']:>13.4f}"
          f"{entry['fixed_p95_seconds'] / entry['cert_p95_seconds']:>12.3f}")
    print(f"{'mean coarse fraction':<24}{entry['cert_coarse']:>13.4f}"
          f"{entry['fixed_coarse']:>13.4f}")
    print(f"\n{'safety / integrity':<24}{'certificate':>13}{'fixed step':>13}")
    print(f"{'unsafe / audited':<24}"
          f"{str(entry['cert_unsafe']) + '/' + str(entry['cert_audited']):>13}"
          f"{str(entry['fixed_unsafe']) + '/' + str(entry['fixed_audited']):>13}")
    print(f"{'worst clearance (m)':<24}{entry['cert_worst_clearance']:>13.4f}"
          f"{entry['fixed_worst_clearance']:>13.4f}")
    print(f"{'self-colliding':<24}{entry['cert_self_colliding']:>13}"
          f"{entry['fixed_self_colliding']:>13}")
    print(f"{'missed edges (must be 0)':<24}{entry['cert_misses']:>13}"
          f"{entry['fixed_misses']:>13}")
    print(f"\npaired: fixed step slower on {entry['paired_slower_without']}/"
          f"{entry['paired_n']} problems")


def per_scene_table(data, out, scenes):
    key = ["seed", "scene", "problem"]
    both = solved(data["cert"]).merge(solved(data["fixed"]), on=key, suffixes=("_c", "_f"))
    rows_out = []
    for scene in scenes:
        pair = both[both["scene"] == scene]
        entry = {"scene": scene, "n": int(len(pair))}
        for column, _, scale, _ in METRICS:
            c = float(pair[f"{column}_c"].median() * scale)
            f = float(pair[f"{column}_f"].median() * scale)
            entry[f"{column}_cert"] = c
            entry[f"{column}_fixed"] = f
            entry[f"{column}_ratio"] = f / c if c else float("nan")
            # Paired rather than a ratio of medians: same problem under both caps, so a
            # scene whose hard problems dominate the median cannot fake a speedup.
            entry[f"{column}_ratio_paired"] = float(
                (pair[f"{column}_f"] / pair[f"{column}_c"].replace(0, np.nan)).median())
        entry["coarse_cert"] = float(pair["coarse_fraction_c"].mean())
        entry["total_ms_cert"] = float(pair["seconds_c"].sum() * 1e3)
        entry["total_ms_fixed"] = float(pair["seconds_f"].sum() * 1e3)
        entry["unsafe_cert"] = int(pair["unsafe_states_c"].sum())
        entry["unsafe_fixed"] = int(pair["unsafe_states_f"].sum())
        entry["audited_cert"] = int(pair["audited_states_c"].sum())
        entry["audited_fixed"] = int(pair["audited_states_f"].sum())
        rows_out.append(entry)

    pd.DataFrame(rows_out).to_csv(out / "cert_ab_per_scene.csv", index=False)

    saved = sum(e["total_ms_fixed"] - e["total_ms_cert"] for e in rows_out)
    budget = sum(e["total_ms_cert"] for e in rows_out)

    print("\nper scene, paired medians; ratio is fixed/cert, >1 = certificate wins\n")
    head = (f"{'scene':<19}{'n':>4}{'ms cert':>9}{'ms fixed':>10}{'ms x':>7}"
            f"{'evals cert':>12}{'evals x':>9}{'vtx cert':>10}{'vtx x':>7}"
            f"{'coarse':>8}{'path x':>8}")
    print(head)
    print("-" * len(head))
    for e in rows_out:
        print(f"{e['scene']:<19}{e['n']:>4}{e['seconds_cert']:>9.3f}"
              f"{e['seconds_fixed']:>10.3f}{e['seconds_ratio']:>7.2f}"
              f"{e['samples_cert']:>12.0f}{e['samples_ratio']:>9.2f}"
              f"{e['vertices_cert']:>10.0f}{e['vertices_ratio']:>7.2f}"
              f"{e['coarse_cert']:>8.1%}{e['path_length_ratio']:>8.3f}")

    print(f"\n{'scene':<19}{'total ms cert':>14}{'total ms fixed':>15}{'saved':>8}"
          f"{'of all saving':>15}{'of total time':>15}")
    for e in rows_out:
        d = e["total_ms_fixed"] - e["total_ms_cert"]
        print(f"{e['scene']:<19}{e['total_ms_cert']:>14.1f}{e['total_ms_fixed']:>15.1f}"
              f"{d:>8.1f}{100 * d / saved:>14.1f}%"
              f"{100 * e['total_ms_cert'] / budget:>14.1f}%")
    print(f"{'ALL':<19}{budget:>14.1f}{budget + saved:>15.1f}{saved:>8.1f}"
          f"{100.0:>14.1f}%{100.0:>14.1f}%")
    print(f"\naggregate time the certificate removes: {100 * saved / (budget + saved):.1f}%")

    print(f"\n{'scene':<19}{'unsafe/audited cert':>22}{'unsafe/audited fixed':>23}")
    for e in rows_out:
        print(f"{e['scene']:<19}"
              f"{str(e['unsafe_cert']) + '/' + str(e['audited_cert']):>22}"
              f"{str(e['unsafe_fixed']) + '/' + str(e['audited_fixed']):>23}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/mbm_cert"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(root) / "plots")
    out.mkdir(parents=True, exist_ok=True)

    data = {mode: load(root, mode) for mode in ("cert", "fixed")}
    scenes = sorted(set(solved(data["cert"])["scene"]))

    pooled(data, out, len(solved(data["cert"])))
    for column in PER_SCENE:
        facet(data, out, column, scenes)
    pooled_table(data, out)
    per_scene_table(data, out, scenes)

    print(f"\nwrote {out}/cert_ab_cdf.png, "
          + ", ".join(f"cert_ab_per_scene_{c}_cdf.png" for c in PER_SCENE)
          + f", cert_ab_summary.csv, cert_ab_per_scene.csv")


if __name__ == "__main__":
    main()
