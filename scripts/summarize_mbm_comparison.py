#!/usr/bin/env python3
"""Aggregate per-problem CSV files emitted by demo_UR5MBMBenchmark."""

import argparse
import csv
import math
import statistics
from pathlib import Path


METHODS = ("isSafe", "bubbleCBF", "VAMP")


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return math.nan
    position = fraction * (len(values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    return values[low] * (high - position) + values[high] * (position - low)


def load(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                for name in ("seed", "problem", "eligible", "solved", "samples", "vertices",
                             "waypoints", "audited_states", "unsafe_states", "self_colliding",
                             "misses"):
                    row[name] = int(row[name])
                for name in ("seconds", "path_length", "min_clearance", "min_self_overlap",
                             "rad_per_call", "coarse_fraction"):
                    row[name] = float(row[name])
                row["audit_safe"] = bool(row["solved"] and row["unsafe_states"] == 0 and
                                         row["self_colliding"] == 0 and row["misses"] == 0)
                rows.append(row)
    return rows


def summarize(rows):
    eligible = [row for row in rows if row["eligible"]]
    solved = [row for row in eligible if row["solved"]]
    safe = [row for row in eligible if row["audit_safe"]]
    seconds = [row["seconds"] * 1000 for row in safe]
    samples = [row["samples"] for row in safe]
    total_seconds = sum(row["seconds"] for row in eligible)
    total_samples = sum(row["samples"] for row in eligible)
    safe_count = len(safe)
    return {
        "eligible_runs": len(eligible),
        "returned_solutions": len(solved),
        "return_rate_pct": 100 * len(solved) / len(eligible) if eligible else math.nan,
        "audit_safe_solutions": len(safe),
        "audit_safe_rate_pct": 100 * len(safe) / len(eligible) if eligible else math.nan,
        "unsafe_returned": len(solved) - len(safe),
        "returned_precision_pct": 100 * len(safe) / len(solved) if solved else math.nan,
        "median_ms_safe": statistics.median(seconds) if seconds else math.nan,
        "p90_ms_safe": percentile(seconds, 0.9),
        "median_samples_safe": statistics.median(samples) if samples else math.nan,
        "p90_samples_safe": percentile(samples, 0.9),
        "median_vertices_safe": statistics.median(row["vertices"] for row in safe) if safe else math.nan,
        "median_path_length_safe": statistics.median(row["path_length"] for row in safe) if safe else math.nan,
        "total_seconds": total_seconds,
        "total_samples": total_samples,
        "safe_solutions_per_second": safe_count / total_seconds if total_seconds else math.nan,
        "amortized_ms_per_safe_solution": 1000 * total_seconds / safe_count if safe_count else math.nan,
        "amortized_samples_per_safe_solution": total_samples / safe_count if safe_count else math.nan,
    }


def paired_ratios(rows, candidate, field):
    keyed = {(row["seed"], row["scene"], row["problem"], row["method"]): row for row in rows}
    ratios = []
    for key, baseline in keyed.items():
        seed, scene, problem, method = key
        if method != "isSafe" or not baseline["audit_safe"] or baseline[field] <= 0:
            continue
        other = keyed.get((seed, scene, problem, candidate))
        if other and other["audit_safe"] and other[field] > 0:
            ratios.append(baseline[field] / other[field])
    return ratios


def write_table(path, rows, fieldnames):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.csv)

    raw_fields = [name for name in rows[0] if name != "audit_safe"] + ["audit_safe"]
    write_table(args.output_dir / "all_runs.csv", rows, raw_fields)

    overall = []
    for method in METHODS:
        summary = {"method": method, **summarize([row for row in rows if row["method"] == method])}
        time_ratios = paired_ratios(rows, method, "seconds") if method != "isSafe" else [1.0]
        sample_ratios = paired_ratios(rows, method, "samples") if method != "isSafe" else [1.0]
        summary["paired_time_speedup_vs_isSafe"] = statistics.median(time_ratios) if time_ratios else math.nan
        summary["paired_sample_reduction_vs_isSafe"] = statistics.median(sample_ratios) if sample_ratios else math.nan
        summary["paired_safe_cases"] = len(time_ratios) if method != "isSafe" else summary["audit_safe_solutions"]
        overall.append(summary)
    write_table(args.output_dir / "overall_summary.csv", overall, list(overall[0]))

    scenes = sorted({row["scene"] for row in rows})
    by_scene = []
    for scene in scenes:
        for method in METHODS:
            selected = [row for row in rows if row["scene"] == scene and row["method"] == method]
            by_scene.append({"scene": scene, "method": method, **summarize(selected)})
    write_table(args.output_dir / "scene_summary.csv", by_scene, list(by_scene[0]))

    seeds = sorted({row["seed"] for row in rows})
    by_seed = []
    for seed in seeds:
        for method in METHODS:
            selected = [row for row in rows if row["seed"] == seed and row["method"] == method]
            by_seed.append({"seed": seed, "method": method, **summarize(selected)})
    write_table(args.output_dir / "seed_summary.csv", by_seed, list(by_seed[0]))

    def number(value, digits=1):
        return "—" if not math.isfinite(value) else f"{value:.{digits}f}"

    lines = [
        "# UR5 MotionBenchMaker comparison",
        "",
        f"Five seeds, 20 problems per scene, seven scenes: {len(rows):,} method rows; "
        f"{sum(row['eligible'] for row in rows):,} eligible method runs.",
        "",
        "Parameters: RRTConnect, 0.5 s timeout, 2.0 rad range, 0.03 m SDF voxel, "
        "0 world margin, 0 self margin, automatic SDF interpolation guard buffer, "
        "0.05 s CBF step, CBF decay rate 8 /s (the old per-step gamma 0.4 at that "
        "step), no shortcutting. "
        "Safety is a 0.02 rad dense audit against the original exact box/cylinder primitives.",
        "The `samples` metric counts checked configurations for isSafe, barrier evaluations "
        "for bubbleCBF, and sampled configurations for VAMP (motion batches include SIMD padding).",
        "",
        "## Overall",
        "",
        "| Method | Returned | Audit-safe | Unsafe returned | Safe time median / p90 (ms) | "
        "Safe samples median / p90 | Vertices median | Path median (rad) | Paired time speedup | "
        "Paired sample reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in overall:
        lines.append(
            f"| {item['method']} | {item['returned_solutions']}/{item['eligible_runs']} "
            f"({item['return_rate_pct']:.1f}%) | {item['audit_safe_solutions']}/{item['eligible_runs']} "
            f"({item['audit_safe_rate_pct']:.1f}%) | {item['unsafe_returned']} | "
            f"{number(item['median_ms_safe'])} / {number(item['p90_ms_safe'])} | "
            f"{number(item['median_samples_safe'], 0)} / {number(item['p90_samples_safe'], 0)} | "
            f"{number(item['median_vertices_safe'], 0)} | {number(item['median_path_length_safe'], 2)} | "
            f"{number(item['paired_time_speedup_vs_isSafe'], 2)}× | "
            f"{number(item['paired_sample_reduction_vs_isSafe'], 2)}× |"
        )

    lines.extend([
        "", "## Failure-aware efficiency", "",
        "These totals include all timeout and rejected-path work, rather than conditioning on success.",
        "",
        "| Method | Audit precision among returned | Safe solutions / wall-second | "
        "Amortized wall time / safe solution (ms) | Amortized samples / safe solution |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in overall:
        lines.append(
            f"| {item['method']} | {item['returned_precision_pct']:.1f}% | "
            f"{number(item['safe_solutions_per_second'], 2)} | "
            f"{number(item['amortized_ms_per_safe_solution'])} | "
            f"{number(item['amortized_samples_per_safe_solution'], 0)} |"
        )

    lines.extend([
        "", "## Audit-safe completion rate by seed", "",
        "| Seed | isSafe | bubbleCBF | VAMP |", "|---:|---:|---:|---:|",
    ])
    seed_map = {(item["seed"], item["method"]): item for item in by_seed}
    for seed in seeds:
        cells = [str(seed)]
        for method in METHODS:
            item = seed_map[(seed, method)]
            cells.append(f"{item['audit_safe_solutions']}/{item['eligible_runs']} "
                         f"({item['audit_safe_rate_pct']:.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "", "## Audit-safe completion rate by scene", "",
        "| Scene | isSafe | bubbleCBF | VAMP |", "|---|---:|---:|---:|",
    ])
    scene_map = {(item["scene"], item["method"]): item for item in by_scene}
    for scene in scenes:
        cells = [scene]
        for method in METHODS:
            item = scene_map[(scene, method)]
            cells.append(f"{item['audit_safe_solutions']}/{item['eligible_runs']} "
                         f"({item['audit_safe_rate_pct']:.1f}%)")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "", "## Median time for audit-safe solutions by scene (ms)", "",
        "| Scene | isSafe | bubbleCBF | VAMP |", "|---|---:|---:|---:|",
    ])
    for scene in scenes:
        cells = [scene] + [number(scene_map[(scene, method)]["median_ms_safe"]) for method in METHODS]
        lines.append("| " + " | ".join(cells) + " |")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
