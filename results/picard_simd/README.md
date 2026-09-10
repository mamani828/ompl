# Where the CBF filter's time actually goes, and what SIMD is worth

This directory records the measurements behind a negative result: SIMD across a Picard
window has nothing to batch, because a window wide enough to be worth batching is a
window that costs more than it commits.

## The harness

`demos/CBFSteerBenchmark.cpp` (`demo_CBFSteerBenchmark <scene.grid> [edges] [seed]
[range] [kappa]`) times the steering primitive on its own: the same edges, drawn once,
steered by each method. The planner-level A/B in `../picard_sliding_benchmark` measures
whole solves, where the methods build different trees and the search's variance rides
along with the steering cost. This separates the two. What it cannot answer is whether
cheaper edges make a faster planner -- a rollout that ends somewhere else is a different
tree -- so it is a kernel measurement, not a verdict.

## Sliding beats converging, decisively

The question SIMD hangs on: does a wide window commit enough to pay for itself? A window
of `w` costs `w` filter calls per map application whether it commits all of them or one.

Measured on `corridor`, `shelf` and `clutter` with `picardConverge 1`, which iterates
each window to a fixed point instead of sliding on the first certified prefix:

| policy | calls per committed step |
| --- | --- |
| slide on the first prefix (current) | 1.05 - 1.08 |
| converge the window, w=4, 2 iterations | 2.94 - 3.49 |
| converge the window, w=8, 2 iterations | 5.19 - 6.49 |
| converge the window, w=8, 4 iterations | 5.30 - 6.62 |

More iterations do not fix it: at four iterations a window of 8 still commits only 4.1
of its 8 segments, so the map is not converging over the window, it is converging over
the first few steps of one. Sliding already sits at 1.05-1.08 calls per committed step
against a floor of 1.0 -- every committed segment needs one filter evaluation at its
start, and that is the safety contract, not an inefficiency.

Perfect four-lane SIMD would price a speculative state at ~0.5 of a scalar call (the
QP is ~30% of a call and does not vectorise: it is a branchy active set whose iteration
count differs per state). Converging windows would then land at 1.3-1.6 calls per step
against sliding's 1.05. **The batch that SIMD would accelerate is work that should not
be done.**

There is a second, sharper reason a wide window cannot pay once the certificate is in
play: a stretched segment ends its window. Running segment `k` past `stepSize` moves
where segment `k + 1` starts, and that segment's control was certified at the state the
map evaluated. Holding the stretch back to the window's last segment so a wide window
could commit itself whole was tried and measured worse by half -- it left the
certificate unspent on every segment but one, and an edge the sequential rollout crossed
in 26.6 coarse steps took 34.7 shorter ones. The certificate and the window width are in
direct competition, and the certificate wins.

## SIMD inside a filter call is worth 2-6%

`GridSDF`'s AVX2 batch kernel is now built and dispatched (one translation unit at
`-mavx2`, a runtime `__builtin_cpu_supports` check, bit-identical to the scalar path).
It changed the steering benchmark by less than the run-to-run noise: the kernel
vectorises the trilinear interpolation, while the cost of an SDF query is the cell
lookup and the eight scattered corner loads, which it does not.

Rebuilding everything at `-march=x86-64-v3`, so Eigen's fixed-size operations go
four-wide throughout:

| scene | sequential baseline | at x86-64-v3 |
| --- | --- | --- |
| corridor, per edge | 51.5 us | 50.3 us |
| clutter, per edge | 75.2 us | 70.5 us |
| MotionBenchMaker, median solve | 0.80 ms | 0.76 ms |

2-6%, and it applies to both steering methods equally, so it changes no ranking. The
303-pair self-collision loop -- the obvious hand-vectorisation target -- already runs at
about four cycles per pair including its square root, which is near scalar peak; AVX2's
`vsqrtpd` throughput would take that to roughly three. The filter is not SIMD-bound. It
is bound by a branchy QP, scattered grid loads, and dependent scalar chains.

## MotionBenchMaker

`mbm_s<seed>_p<0|2>.csv`: 140 problems, seeds 1-3, 5 s limit, kappa 8/s, `p0` the
sequential rollout and `p2` speculative steering at 2 iterations with adaptive windows.
420 problem/seed pairs, all solved by both.

Paired geometric mean, Picard against sequential: **1.103x** wall time. Picard is faster
on 95 of 420. By scene: `table_under_pick` 1.00x, `bookshelf_small` 1.05x,
`bookshelf_thin` 1.08x, `box` 1.10x, `table_pick` 1.16x, `bookshelf_tall` 1.16x, `cage`
1.19x.

The gap does narrow in the constrained scenes -- 1.10x here against 1.37x on the
PyBullet suite, whose geometric mean is dominated by scenes solved in 0.1 ms where fixed
overheads are the whole story; on the two expensive PyBullet scenes it is 1.05x
(`corridor`) and 1.12x (`shelf`). It does not cross 1.

One thing to watch rather than dismiss: audited states below the margin came to
1860/294512 for Picard against 1541/293665 for sequential over the same 420 pairs. Both
are non-zero before either method is involved, the worst clearance is the *better* of
the two for Picard (-0.0097 m against -0.0122 m), and the trees differ -- but the 21%
gap in count is not explained by anything in the steering rule and has not been chased
down.

`pybullet_ab.csv` is the matching planner-level A/B on the five PyBullet scenes.
