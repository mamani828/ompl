# UR5 MotionBenchMaker comparison

3 seed(s), 7 scene(s): 2,100 method rows; 2,100 eligible method runs.

Run parameters are not encoded in the CSV; preserve the benchmark invocation alongside this report. Safety fields come from the benchmark's dense audit against the original exact box/cylinder primitives.
The `samples` metric counts checked configurations for isSafe, barrier evaluations for qpFixed/bubbleCBF/qpFreeGate, and sampled configurations for VAMP (motion batches include SIMD padding).

## Overall

| Method | Returned | Audit-safe | Unsafe returned | Safe time median / p90 (ms) | Safe samples median / p90 | Vertices median | Path median (rad) | Paired time speedup | Paired sample reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isSafe | 420/420 (100.0%) | 252/420 (60.0%) | 168 | 1.4 / 36.9 | 1571 / 41116 | 19 | 13.11 | 1.00× | 1.00× |
| qpFixed | 420/420 (100.0%) | 244/420 (58.1%) | 176 | 0.5 / 6.5 | 308 / 3609 | 10 | 8.27 | 2.12× | 3.72× |
| bubbleCBF | 420/420 (100.0%) | 246/420 (58.6%) | 174 | 0.6 / 7.3 | 298 / 3225 | 11 | 9.16 | 2.94× | 6.74× |
| qpFreeGate | 240/420 (57.1%) | 240/420 (57.1%) | 0 | 2.1 / 59.8 | 1984 / 49389 | 50 | 14.92 | 0.80× | 0.97× |
| VAMP | 420/420 (100.0%) | 223/420 (53.1%) | 197 | 0.1 / 1.0 | 2915 / 39643 | 16 | 12.31 | 10.81× | 0.43× |

## Failure-aware efficiency

These totals include all timeout and rejected-path work, rather than conditioning on success.

| Method | Audit precision among returned | Safe solutions / wall-second | Amortized wall time / safe solution (ms) | Amortized samples / safe solution |
|---|---:|---:|---:|---:|
| isSafe | 60.0% | 32.53 | 30.7 | 28599 |
| qpFixed | 58.1% | 281.83 | 3.5 | 1984 |
| bubbleCBF | 58.6% | 256.82 | 3.9 | 1721 |
| qpFreeGate | 100.0% | 0.27 | 3769.4 | 1123213 |
| VAMP | 53.1% | 207.80 | 4.8 | 116152 |

## Audit-safe completion rate by seed

| Seed | isSafe | qpFixed | bubbleCBF | qpFreeGate | VAMP |
|---:|---:|---:|---:|---:|---:|
| 1 | 77/140 (55.0%) | 87/140 (62.1%) | 85/140 (60.7%) | 80/140 (57.1%) | 71/140 (50.7%) |
| 2 | 86/140 (61.4%) | 77/140 (55.0%) | 81/140 (57.9%) | 80/140 (57.1%) | 71/140 (50.7%) |
| 3 | 89/140 (63.6%) | 80/140 (57.1%) | 80/140 (57.1%) | 80/140 (57.1%) | 81/140 (57.9%) |

## Audit-safe completion rate by scene

| Scene | isSafe | qpFixed | bubbleCBF | qpFreeGate | VAMP |
|---|---:|---:|---:|---:|---:|
| bookshelf_small | 38/60 (63.3%) | 19/60 (31.7%) | 25/60 (41.7%) | 30/60 (50.0%) | 32/60 (53.3%) |
| bookshelf_tall | 32/60 (53.3%) | 31/60 (51.7%) | 30/60 (50.0%) | 21/60 (35.0%) | 29/60 (48.3%) |
| bookshelf_thin | 32/60 (53.3%) | 25/60 (41.7%) | 24/60 (40.0%) | 9/60 (15.0%) | 29/60 (48.3%) |
| box | 56/60 (93.3%) | 53/60 (88.3%) | 51/60 (85.0%) | 60/60 (100.0%) | 52/60 (86.7%) |
| cage | 40/60 (66.7%) | 38/60 (63.3%) | 38/60 (63.3%) | 60/60 (100.0%) | 25/60 (41.7%) |
| table_pick | 21/60 (35.0%) | 46/60 (76.7%) | 43/60 (71.7%) | 33/60 (55.0%) | 23/60 (38.3%) |
| table_under_pick | 33/60 (55.0%) | 32/60 (53.3%) | 35/60 (58.3%) | 27/60 (45.0%) | 33/60 (55.0%) |

## Median time for audit-safe solutions by scene (ms)

| Scene | isSafe | qpFixed | bubbleCBF | qpFreeGate | VAMP |
|---|---:|---:|---:|---:|---:|
| bookshelf_small | 1.6 | 0.3 | 0.2 | 1.7 | 0.1 |
| bookshelf_tall | 2.2 | 0.7 | 0.9 | 1.5 | 0.1 |
| bookshelf_thin | 2.2 | 0.3 | 0.3 | 1.6 | 0.2 |
| box | 0.7 | 0.5 | 0.5 | 1.0 | 0.1 |
| cage | 42.3 | 7.4 | 8.6 | 48.0 | 2.3 |
| table_pick | 1.0 | 0.3 | 0.1 | 2.4 | 0.1 |
| table_under_pick | 0.7 | 1.1 | 1.2 | 1.2 | 0.1 |
