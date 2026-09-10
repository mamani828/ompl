#!/usr/bin/env python3
"""Summarize a merged demo_UR5MBMBenchmark CSV as box-drawn tables.

Every row-vs-row number is *paired* -- computed only over problems both rows solved.
An unpaired per-row median is not a comparison when the rows solve different
problem sets, which they do here: `qp-free` lands 519 where the others land 520,
and a config change can move that by tens of problems.

    ./scripts/summarize_mbm_runs.py <merged.csv>
"""
import csv
import statistics
import sys
from collections import defaultdict

LABEL = [("isSafe", "rrtconnect"), ("qpFixed", "qp-fixed"), ("qpAdaptive", "qp-adapt (L1)"),
         ("qpFreeGate", "qp-free (no QP)"), ("qpEnvelope", "qp-env")]

PAIRS = [("qpAdaptive", "qpEnvelope"), ("qpFixed", "qpAdaptive"), ("isSafe", "qpEnvelope"),
         ("isSafe", "qpAdaptive"), ("qpFreeGate", "qpEnvelope")]

NAME = dict(LABEL)


def med(values):
    return statistics.median(values) if values else float("nan")


def table(headers, rows, aligns=None, rule_between=True):
    """Box-drawn table. \\p aligns is one of 'l' / 'r' / 'c' per column."""
    aligns = aligns or ["l"] + ["r"] * (len(headers) - 1)
    width = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
             for i, h in enumerate(headers)]

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in width) + right

    def row(cells, centre=False):
        out = []
        for i, cell in enumerate(cells):
            text = str(cell)
            if centre:
                out.append(text.center(width[i]))
            elif aligns[i] == "l":
                out.append(text.ljust(width[i]))
            elif aligns[i] == "c":
                out.append(text.center(width[i]))
            else:
                out.append(text.rjust(width[i]))
        return "│ " + " │ ".join(out) + " │"

    lines = [line("┌", "┬", "┐"), row(headers, centre=True), line("├", "┼", "┤")]
    for index, r in enumerate(rows):
        if index and rule_between:
            lines.append(line("├", "┼", "┤"))
        lines.append(row(r))
    lines.append(line("└", "┴", "┘"))
    return "\n".join(lines)


def signed(value):
    """Clearance with an explicit sign, so a negative one cannot be skimmed past."""
    if value != value:
        return "-"
    return f"{'−' if value < 0 else '+'}{abs(value):.4f}"


def main(path):
    by_problem = defaultdict(dict)
    with open(path, newline="") as stream:
        for record in csv.DictReader(stream):
            by_problem[(record["scene"], record["problem"])][record["method"]] = record

    rows = defaultdict(list)
    for entry in by_problem.values():
        for method, record in entry.items():
            rows[method].append(record)

    scenes = sorted({scene for scene, _ in by_problem})
    print(f"\n{len(by_problem)} problems over {len(scenes)} scenes: {', '.join(scenes)}\n")

    body = []
    for method, label in LABEL:
        records = rows.get(method)
        if not records:
            continue
        ok = [r for r in records if r["solved"] == "1"]
        unsafe = sum(int(r["unsafe_states"]) for r in records)
        audited = sum(int(r["audited_states"]) for r in records)
        clearance = [float(r["min_clearance"]) for r in records
                     if float(r["min_clearance"]) == float(r["min_clearance"])
                     and abs(float(r["min_clearance"])) != float("inf")]
        dirty = sum(1 for r in ok if int(r["unsafe_states"]) or int(r["self_colliding"])
                    or int(r["misses"]))
        body.append([
            label,
            f"{len(ok)}/{len(records)}",
            f"{med([1e3 * float(r['seconds']) for r in ok]):.2f}",
            f"{med([float(r['samples']) for r in ok]):,.0f}",
            f"{med([float(r['vertices']) for r in ok]):.0f}",
            f"{med([float(r['path_length']) for r in ok if float(r['path_length']) > 0]):.2f}",
            f"{unsafe:,}/{audited:,}",
            signed(min(clearance, default=float('nan'))),
            f"{dirty}/{len(ok)}",
        ])
    print(table(["row", "solved", "med ms", "evaluations", "vertices", "path",
                 "unsafe/audited", "worst clr", "dirty"], body))

    every = [k for k, v in by_problem.items()
             if all(m in v and v[m]["solved"] == "1" for m, _ in LABEL)]
    print(f"\nsolved by all five rows: {len(every)}\n")

    print("paired comparisons -- only problems both rows solved; change is to-vs-from,\n"
          "so a negative number means the second row needed less\n")
    paired = []
    for a, b in PAIRS:
        both = [k for k, v in by_problem.items()
                if all(m in v and v[m]["solved"] == "1" for m in (a, b))]
        if not both:
            continue
        cells = [f"{NAME[a]} → {NAME[b]}", len(both)]
        for column, factor, fmt in (("samples", 1, ",.0f"), ("seconds", 1e3, ".2f"),
                                    ("vertices", 1, ".0f"), ("path_length", 1, ".2f")):
            x = med([factor * float(by_problem[k][a][column]) for k in both])
            y = med([factor * float(by_problem[k][b][column]) for k in both])
            delta = 1e2 * (y - x) / x if x else float("nan")
            cells.append(f"{format(x, fmt)} → {format(y, fmt)}  {delta:+.1f}%")
        ratios = sorted(float(by_problem[k][b]["samples"]) / float(by_problem[k][a]["samples"])
                        for k in both if float(by_problem[k][a]["samples"]) > 0)
        cells.append(f"{sum(1 for r in ratios if r < 1)}/{len(ratios)}")
        cells.append(f"{med(ratios):.3f}")
        paired.append(cells)
    print(table(["comparison", "n", "evaluations", "med ms", "vertices", "path",
                 "2nd lower on", "ratio"], paired,
                aligns=["l", "r", "r", "r", "r", "r", "r", "r"]))


if __name__ == "__main__":
    main(sys.argv[1])
