#!/usr/bin/env python3
"""Paired UR5 bound ablation, with self-collision enabled and fixed planner settings.

Run after building demo_UR5MBMBenchmark. Alternates order across seeds and saves
raw CSVs/logs; timings include planning only (as reported by the benchmark).
"""
import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build/demos/demo_UR5MBMBenchmark")
    parser.add_argument("--scenes", default="scenes.txt")
    parser.add_argument("--output", default="results/lipschitz_bounds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--per-scene", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {mode: [] for mode in ("arm_length", "tight")}
    for index, seed in enumerate(args.seeds):
        modes = list(rows) if index % 2 == 0 else list(reversed(rows))
        for mode in modes:
            path = output / f"{mode}_{seed}.csv"
            env = dict(os.environ, OMPL_UR5_LEVER_BOUNDS=mode)
            command = [args.binary, args.scenes, str(args.per_scene), "1.0", "0.02", "0.05",
                       "2.0", "0.002", "0.006", "-1", "8", "-1", "0", "", "-1",
                       str(seed), str(path), "1"]
            print(f"Running {mode}, seed {seed}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            with path.open() as data:
                rows[mode].extend(r for r in csv.DictReader(data)
                                  if r["method"] == "bubbleCBF" and r["eligible"] == "1")
    for mode, data in rows.items():
        solved = [r for r in data if r["solved"] == "1"]
        print(f"{mode}: solved {len(solved)}/{len(data)}; "
              f"median ms={1000 * statistics.median(float(r['seconds']) for r in solved):.4f}; "
              f"median calls={statistics.median(int(r['samples']) for r in solved):.1f}; "
              f"median rad/call={statistics.median(float(r['rad_per_call']) for r in solved):.5f}; "
              f"unsafe states={sum(int(r['unsafe_states']) for r in solved)}; "
              f"self-colliding states={sum(int(r['self_colliding']) for r in solved)}")
    keyed = [{(r["seed"], r["scene"], r["problem"]): r for r in rows[mode]} for mode in rows]
    pairs = [(r, keyed[1][key]) for key, r in keyed[0].items()
             if key in keyed[1] and r["solved"] == keyed[1][key]["solved"] == "1"]
    print(f"Paired solved problems: {len(pairs)}; median baseline/tight time="
          f"{statistics.median(float(a['seconds']) / float(b['seconds']) for a, b in pairs):.3f}x; "
          f"median baseline/tight calls="
          f"{statistics.median(int(a['samples']) / int(b['samples']) for a, b in pairs):.3f}x")


if __name__ == "__main__":
    main()
