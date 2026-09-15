#!/usr/bin/env python3
"""Per-scene LaTeX tables from a merged demo_UR5MBMBenchmark CSV.

    ./scripts/latex_mbm_scene_table.py merged.csv [metric]

`metric` is `solved` (default), `evaluations`, `time`, `vertices` or `qprows`. Medians are over
each row's own solved problems in that scene, so a scene where one row solves far fewer
problems is not comparing the same set -- the solved table is what says when that
happens, and it should be read alongside any of the others.
"""
import csv, statistics, sys
from collections import defaultdict

LABEL = [("isSafe", r"RRT-Connect"), ("qpPlain", r"CBF-RRT"),
         ("qpFixed", r"Adaptive (Fixed)"), ("qpAdaptive", r"Adaptive (Lipschitz)"),
         ("qpFreeGate", r"CBF-RRT (No QP)"), ("qpEnvelope", r"Adaptive (Envelope)")]
PRETTY = {"bookshelf_small": "Small Bookshelf", "bookshelf_tall": "Tall Bookshelf",
          "bookshelf_thin": "Narrow Bookshelf", "box": "Box", "cage": "Cage",
          "table_pick": "Tabletop Pick", "table_under_pick": "Under-Table Pick"}
COLUMN = {"evaluations": ("samples", 1, ",.0f"), "time": ("seconds", 1e3, ".1f"),
          "vertices": ("vertices", 1, ".0f")}


def med(values):
    return statistics.median(values) if values else float("nan")


def metric_cell(metric, records):
    """One table cell: the metric over these solved rows, or `--` when there is none.

    `qprows` is the constraint-row count of the QP the filter assembled per call.
    `qp_rows` is summed over calls by the benchmark, so the per-problem program size is
    the ratio, and the median is taken over problems rather than pooled over calls -- a
    single hard problem makes orders of magnitude more calls than an easy one. Rows that
    build no QP (`isSafe`, `qpFreeGate`) leave the counters at zero and print `--`, as
    does a CSV written before the counters existed.
    """
    if metric == "solved":
        return str(len(records))
    if metric == "qprows":
        if not records or "qp_calls" not in records[0]:
            return "--"
        ratios = [int(r["qp_rows"]) / int(r["qp_calls"]) for r in records
                  if int(r["qp_calls"]) > 0]
        return format(med(ratios), ".0f") if ratios else "--"
    column, factor, fmt = COLUMN[metric]
    return format(med([factor * float(r[column]) for r in records]), fmt) if records else "--"


def main(path, metric="solved"):
    rows = list(csv.DictReader(open(path, newline="")))
    present = [(m, lab) for m, lab in LABEL if any(r["method"] == m for r in rows)]
    eligible = defaultdict(set)
    solved = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["eligible"] == "1":
            eligible[r["scene"]].add(r["problem"])
        if r["solved"] == "1":
            solved[r["scene"]][r["method"]].append(r)
    scenes = sorted(eligible)

    caption = {"solved": "Problems solved per scene, out of the eligible count.",
               "evaluations": "Median filter evaluations (collision checks for RRT-Connect).",
               "time": "Median planning time in milliseconds.",
               "vertices": "Median tree vertices.",
               "qprows": "Median QP constraint rows assembled per filter call; "
                         "rows that solve no QP are marked --."}[metric]

    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(f"  \\caption{{{caption} Medians are over each row's own solved problems.}}")
    print(r"  \label{tab:mbm-per-scene-" + metric + r"}")
    print(r"  \small")
    print(r"  \begin{tabular}{l" + "r" * (len(present) + 1) + r"}")
    print(r"    \toprule")
    print(r"    Scene & Elig. & " + " & ".join(lab for _, lab in present) + r" \\")
    print(r"    \midrule")
    for scene in scenes:
        cells = [metric_cell(metric, solved[scene][m]) for m, _ in present]
        print(f"    {PRETTY.get(scene, scene)} & {len(eligible[scene])} & "
              + " & ".join(cells) + r" \\")
    print(r"    \midrule")
    totals = [metric_cell(metric, [r for scene in scenes for r in solved[scene][m]])
              for m, _ in present]
    print(r"    \textbf{All} & " + str(sum(len(v) for v in eligible.values())) + " & "
          + " & ".join(totals) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "solved")
