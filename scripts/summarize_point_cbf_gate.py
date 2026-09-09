#!/usr/bin/env python3
"""Summarize paired point_cbf_rrt_benchmark CSV output using the stdlib."""

import csv
import math
import statistics
import sys


def median(values):
    return statistics.median(values) if values else math.nan


def main(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    modes = ("qp_lipschitz", "qp_free_gate")
    by_mode = {mode: [r for r in rows if r["mode"] == mode] for mode in modes}
    print("mode,solved/trials,median_ms,median_vertices,median_filter_calls,median_path_length,min_clearance,unsafe_samples")
    for mode in modes:
        records = by_mode[mode]
        solved = [r for r in records if int(r["solved"])]
        print(
            f"{mode},{len(solved)}/{len(records)},"
            f"{median([float(r['planning_ms']) for r in solved]):.6g},"
            f"{median([int(r['vertices']) for r in solved]):.6g},"
            f"{median([int(r['filter_calls']) for r in solved]):.6g},"
            f"{median([float(r['path_length']) for r in solved]):.6g},"
            f"{min((float(r['min_clearance']) for r in solved), default=math.nan):.6g},"
            f"{sum(int(r['unsafe_audit_samples']) for r in records)}"
        )

    indexed = {mode: {r["seed"]: r for r in by_mode[mode] if int(r["solved"])} for mode in modes}
    matched = sorted(set(indexed[modes[0]]) & set(indexed[modes[1]]), key=int)
    time_ratios = [
        float(indexed["qp_free_gate"][seed]["planning_ms"])
        / float(indexed["qp_lipschitz"][seed]["planning_ms"])
        for seed in matched
    ]
    call_ratios = [
        int(indexed["qp_free_gate"][seed]["filter_calls"])
        / int(indexed["qp_lipschitz"][seed]["filter_calls"])
        for seed in matched
    ]
    print(f"matched_solved_pairs,{len(matched)}")
    print(f"median_gate_over_qp_time,{median(time_ratios):.6g}")
    print(f"median_gate_over_qp_filter_calls,{median(call_ratios):.6g}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULTS.csv")
    main(sys.argv[1])
