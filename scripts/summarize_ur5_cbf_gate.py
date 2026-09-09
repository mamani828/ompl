#!/usr/bin/env python3
"""Aggregate one or more UR5 paired-benchmark CSV files."""

import csv
import math
import statistics
import sys


def median(values):
    return statistics.median(values) if values else math.nan


def main(paths):
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["_runset"] = path
                rows.append(row)

    modes = ("qp_lipschitz", "qp_free_gate")
    print(
        "method,solved/attempted,median_workload_ms,median_success_ms,"
        "median_filter_calls,median_vertices,median_path_length,unsafe_states,"
        "self_colliding,replay_misses"
    )
    for mode in modes:
        records = [row for row in rows if row["method"] == mode]
        solved = [row for row in records if int(row["solved"])]
        print(
            f"{mode},{len(solved)}/{len(records)},"
            f"{median([1e3 * float(row['seconds']) for row in records]):.6g},"
            f"{median([1e3 * float(row['seconds']) for row in solved]):.6g},"
            f"{median([int(row['filter_calls']) for row in records]):.6g},"
            f"{median([int(row['vertices']) for row in records]):.6g},"
            f"{median([float(row['path_length']) for row in solved]):.6g},"
            f"{sum(int(row['unsafe_states']) for row in records)},"
            f"{sum(int(row['self_colliding']) for row in records)},"
            f"{sum(int(row['replay_misses']) for row in records)}"
        )

    paired = {}
    for row in rows:
        key = (row["_runset"], row["seed"], row["scene"], row["problem"])
        paired.setdefault(key, {})[row["method"]] = row
    both = qp_only = gate_only = neither = 0
    matched_ratios = []
    for pair in paired.values():
        if not all(mode in pair for mode in modes):
            continue
        qp = bool(int(pair["qp_lipschitz"]["solved"]))
        gate = bool(int(pair["qp_free_gate"]["solved"]))
        if qp and gate:
            both += 1
            matched_ratios.append(
                float(pair["qp_free_gate"]["seconds"])
                / float(pair["qp_lipschitz"]["seconds"])
            )
        elif qp:
            qp_only += 1
        elif gate:
            gate_only += 1
        else:
            neither += 1
    print(
        f"paired_outcomes,both={both},qp_only={qp_only},"
        f"gate_only={gate_only},neither={neither}"
    )
    print(f"median_gate_over_qp_time_on_both_solved,{median(matched_ratios):.6g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT.csv [RESULT.csv ...]")
    main(sys.argv[1:])
