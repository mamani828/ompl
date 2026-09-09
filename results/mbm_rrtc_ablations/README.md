# UR5 MotionBenchMaker RRT-Connect ablations

This directory contains five paired seeds for two independent ablations:

1. **QP solve versus no QP**, both using a fixed controller step.
2. **Adaptive versus fixed hold**, both using the same CBF-QP controller.

All rows use RRT-Connect, identical problem/seed pairs, a 1 s planning limit,
5 ms controller steps, a 12 mm SDF guard, no added audited clearance margin,
and a 400-call per-extension budget. The no-QP row implements the direct
barrier test from LQR-CBF-RRT*: evaluate the nominal extension, append accepted
states to the tree, and stop at the first violated CBF constraint.

The 30 mm SDF grid and 12 mm guard admit 10 problems (five box and five cage);
the five seeds therefore give 50 paired runs. The other five scene families
are excluded because their supplied start/goal clearances are below the guard.

## Results

Metrics are measured at the first exact solution. Cost is reported as barrier
evaluations and tree vertices. Medians use solved runs.

| RRT-Connect method | Success | Median time | Median barrier evaluations | Median vertices | Unsafe / audited | Missed edges |
|---|---:|---:|---:|---:|---:|---:|
| QP, fixed hold | 50/50 (100%) | 14.89 ms | 8,182.5 | 30 | 0 / 135,398 | 0 |
| No QP, fixed step | 50/50 (100%) | 70.36 ms | 63,728.5 | 270 | 0 / 131,269 | 0 |
| QP, adaptive hold | 50/50 (100%) | 16.09 ms | 6,586.0 | 30 | 0 / 107,517 | 0 |
| VAMP RRT-Connect baseline | 50/50 (100%) | 0.49 ms | 19,054.5 SIMD lanes | 78.5 | 0 / 39,784 | 0 |
| Scalar collision-checked RRT-Connect | 50/50 (100%) | 24.84 ms | 29,059.5 checks | 76 | 35 / 39,414 | 0 |

The scalar collision-checked baseline is not safety-clean at its sampled edge
resolution and is included only as a timing reference. VAMP is the clean
collision-checking baseline.

On all 50 jointly solved paired runs:

| Ablation | Median time ratio | Median evaluation ratio | Median vertex ratio |
|---|---:|---:|---:|
| No QP / QP, fixed hold | 4.721x | 7.139x | 7.957x |
| QP adaptive / QP fixed | 1.025x | 0.779x | 1.000x |

The adaptive hold removes about 22% of barrier evaluations, but its additional
certificate work makes median wall time statistically flat in this sample.
The no-QP gate avoids optimization per step, but rejecting rather than repairing
nominal controls causes substantially more RRT exploration.

Supplemental certificate rows are also present in the CSVs. holdNew improves
success over l1Old from 64% to 82%, but those rows have 12 and 13 missing
recorded edges respectively, so they are not part of the clean ablation claims.

## Raw data

The paired CSVs are final_s1.csv through final_s5.csv. Seeds are 31 through 35.
The generated trajectory files remain local because they are large.
