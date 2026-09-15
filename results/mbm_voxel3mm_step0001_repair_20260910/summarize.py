#!/usr/bin/env python3
"""Summarize the merged shard CSV from this run.

Kept beside the run rather than in scripts/ because it answers a question the
existing summarizer does not: when rows solve *different* problems, a per-row
median is not a comparison. Every row-vs-row number here is therefore paired --
computed only over problems both rows solved -- and the unpaired table reports
each row's own solved count so the mismatch is visible rather than hidden.
"""
import csv, statistics, sys
from collections import defaultdict

LAB = {"isSafe": "rrtconnect", "qpFixed": "qp-fixed", "qpAdaptive": "qp-adapt (L1)",
       "qpFreeGate": "qp-free (no QP)", "qpEnvelope": "qp-env"}
ORDER = list(LAB)

def med(v):
    return statistics.median(v) if v else float("nan")

def main(path):
    by_problem = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by_problem[(r["scene"], r["problem"])][r["method"]] = r
    rows = defaultdict(list)
    for v in by_problem.values():
        for m, r in v.items():
            rows[m].append(r)

    scenes = sorted({s for s, _ in by_problem})
    print(f"{len(by_problem)} problems over {len(scenes)} scenes: {', '.join(scenes)}\n")

    hdr = (f"{'row':<17}{'solved':>10}{'med ms':>10}{'evaluations':>13}{'vertices':>10}"
           f"{'path':>8}{'unsafe/audited':>20}{'worst clr':>11}{'dirty':>8}")
    print(hdr); print("-" * len(hdr))
    for m in ORDER:
        rs = rows.get(m)
        if not rs:
            continue
        ok = [r for r in rs if r["solved"] == "1"]
        unsafe = sum(int(r["unsafe_states"]) for r in rs)
        audited = sum(int(r["audited_states"]) for r in rs)
        clr = [float(r["min_clearance"]) for r in rs
               if float(r["min_clearance"]) == float(r["min_clearance"])
               and abs(float(r["min_clearance"])) != float("inf")]
        dirty = sum(1 for r in ok if int(r["unsafe_states"]) or int(r["self_colliding"])
                    or int(r["misses"]))
        print(f"{LAB[m]:<17}{len(ok):>5}/{len(rs):<4}"
              f"{med([1e3 * float(r['seconds']) for r in ok]):>10.2f}"
              f"{med([float(r['samples']) for r in ok]):>13,.0f}"
              f"{med([float(r['vertices']) for r in ok]):>10.0f}"
              f"{med([float(r['path_length']) for r in ok if float(r['path_length']) > 0]):>8.2f}"
              f"{unsafe:>9,}/{audited:<10,}{min(clr, default=float('nan')):>11.4f}"
              f"{f'{dirty}/{len(ok)}':>8}")

    every = [k for k, v in by_problem.items()
             if all(m in v and v[m]["solved"] == "1" for m in ORDER)]
    print(f"\nsolved by all five rows: {len(every)}"
          + ("  -- no five-way comparison is possible" if not every else ""))

    print("\npaired comparisons (only problems both rows solved)")
    for a, b in (("qpAdaptive", "qpEnvelope"), ("isSafe", "qpEnvelope"),
                 ("isSafe", "qpAdaptive"), ("qpFixed", "qpAdaptive")):
        both = [k for k, v in by_problem.items()
                if all(m in v and v[m]["solved"] == "1" for m in (a, b))]
        if not both:
            print(f"\n  {LAB[a]} vs {LAB[b]}: no problem solved by both")
            continue
        print(f"\n  {LAB[a]} vs {LAB[b]}  (n = {len(both)})")
        for name, col, scale in (("evaluations", "samples", 1), ("median ms", "seconds", 1e3),
                                 ("vertices", "vertices", 1), ("path length", "path_length", 1)):
            x = med([scale * float(by_problem[k][a][col]) for k in both])
            y = med([scale * float(by_problem[k][b][col]) for k in both])
            print(f"    {name:<14}{x:>12,.2f}{y:>12,.2f}{1e2 * (x - y) / x:>9.2f}%")
        rat = sorted(float(by_problem[k][b]["samples"]) / float(by_problem[k][a]["samples"])
                     for k in both if float(by_problem[k][a]["samples"]) > 0)
        if rat:
            print(f"    per-problem ratio: median {med(rat):.3f} "
                  f"[{rat[0]:.3f}, {rat[-1]:.3f}], "
                  f"{LAB[b]} lower on {sum(1 for v in rat if v < 1)}/{len(rat)}")

if __name__ == "__main__":
    main(sys.argv[1])
