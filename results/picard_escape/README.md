# Tangential escape: what a pointwise rule can and cannot recover

`CBFControlFilter::Parameters::escapeProgress` re-solves the QP against the nominal's
component along the tightest active barrier's level set whenever the projection keeps
less than that share of the nominal. Default zero, which leaves the filter a pure
projection.

This is legal for free. The safety statement is the row `dh_i/dq u >= -kappa h_i` plus
the control box; minimum deviation from the nominal is a *preference* over the feasible
set, not part of the guarantee. `FilteredStateSpace`'s verification is already a
feasibility test rather than an optimality one -- it projects the proposed control and
accepts it if the projection is the identity -- so a re-aimed control certifies through
the existing machinery unchanged.

## The failure it addresses

Not infeasibility. With `h > 0` the row permits approaching an obstacle at up to
`kappa h`, so a nominal pointing into a surface is projected onto a *creep* toward it
that decays exponentially as `h` shrinks. The QP never fails, the step is never unsafe,
and the rollout stops advancing. It shows up as neither `blocked` nor `stalled`.

## Finding the population

`demo_CBFSteerBenchmark`'s `mode` argument. Uniform starts almost never put a sphere in
contact, so a uniform probe measures the wrong population and reports the problem does
not exist:

| probe | stuck (<=5% of the gap closed) |
| --- | --- |
| mode 0, uniform starts and directions | 1 in 1000 |
| mode 1, starts under 20 mm clearance, uniform directions | 0 in 1000 |
| mode 2, those starts, aimed down the tightest barrier's gradient | 195 in 1000 (clutter) |

An RRT's tree nodes concentrate where mode 1 and 2 sample; the extension direction is
a random sample, so the real population sits between them.

## What the escape recovers

1000 edges of 1.5 rad, `escapeProgress` 0.25 against 0.

| scene / probe | travel | gap closed | stuck | ended further away | us/edge |
| --- | --- | --- | --- | --- | --- |
| clutter, mode 2, off | 0.763 rad | 44.6% | 195 | 42 | 92.8 |
| clutter, mode 2, on | 1.152 | 49.9% | 152 | 57 | 108.6 |
| corridor, mode 2, off | 0.878 | 66.6% | 98 | 16 | 58.3 |
| corridor, mode 2, on | 1.157 | 68.6% | 83 | 30 | 66.3 |
| clutter, mode 1, off | 1.315 | 78.1% | 0 | 0 | 94.8 |
| clutter, mode 1, on | 1.328 | 78.2% | 0 | 0 | 79.1 |
| corridor, mode 1, off | 1.351 | 85.2% | 0 | 0 | 51.7 |
| corridor, mode 1, on | 1.369 | 85.3% | 0 | 0 | 53.7 |

Inert where nothing is stuck, which is what a targeted fix should look like. Where things
are stuck it does exactly half the job: **travel rises 32-51% and stuck falls 15-22%,
while the number of extensions ending *further* from their target rises 36-88%.** The
gap actually closed moves 2-5 points. The robot moves; it does not reliably move
usefully.

That split is the point, and it is not a tuning failure -- 0.10, 0.25 and 0.50 all land
within a point of each other on gap closed. It is a statement about what a pointwise rule
knows. A tangential direction has a sign, and choosing it requires knowing where the
trajectory ends up, which is not available at a single state. Head-on the nominal has no
component along the surface at all, so there is no first-order information to choose with
and the implementation takes a deterministic tangent (the joint the tightest barrier
responds to least) -- right about as often as it is wrong.

## What this decides

The lookahead is load-bearing. Escaping deadlock is worth roughly a 20% reduction in
stuck extensions on its own; choosing the escape *direction* is the other half, and it
needs a window objective -- minimise `||q_W - to||` over the window subject to the same
rows -- rather than a per-node preference. That is the one formulation where a window
buys something pointwise steering cannot replicate, as against speculation, which
`../picard_simd/README.md` shows computes the identical trajectory.

Not yet measured: whether enabling the escape helps or hurts an actual planner. More
travel per extension and more extensions landing further away are both real, and which
dominates in a tree is not predictable from these numbers.

## In an actual planner

Both demos take the escape as a trailing argument -- `demo_UR5PyBulletScene`'s
`escapeProgress` (arg 24) and `demo_UR5MBMBenchmark`'s (arg 22). It is a filter-level
change, so it moves the sequential rollout and Picard together; these runs use the
sequential rollout.

**MotionBenchMaker**, `esc_s<seed>_e<0.00|0.25>.csv`, 140 problems x seeds 1-3, all 420
solved by both arms:

| | median ms | median samples | median vertices | median path |
| --- | --- | --- | --- | --- |
| escape off | 0.911 | 348 | 13 | 9.32 |
| escape on | 0.935 | 332 | 12 | 9.28 |

Paired geometric means, on against off: **0.950x samples** (172/420 lower), 0.984x path
length, 1.001x wall time. **PyBullet**, `pybullet_ab.csv`, 48 pairs solved by both:
0.900x filter calls, 0.979x time, identical solve rate.

So the sample-efficiency claim is real and small: 5% fewer samples on MotionBenchMaker,
10% fewer filter calls on the PyBullet suite, at wall time that does not move. Escaping
deadlock does convert some extensions that were creeping into extensions that arrive.

## The cost, which is in the audit

| seed | off: unsafe/audited, worst clearance | on |
| --- | --- | --- |
| 1 | 631/99452, 0.634%, -0.0122 m | 582/97560, 0.597%, -0.0126 m |
| 2 | 411/96428, 0.426%, -0.0112 m | 515/94236, 0.547%, -0.0213 m |
| 3 | 499/97785, 0.510%, -0.0075 m | 573/96664, 0.593%, -0.0108 m |
| all | 1541/293665, 0.525%, -0.0122 m | 1670/288460, 0.579%, -0.0213 m |

Two of three seeds worse on rate, three of three worse on the deepest incursion, which
nearly doubles overall. That is a trend, not counting noise, and there is a mechanism for
it: the escape drives at the *full* nominal speed along the surface. Tangential motion
preserves `h` to first order, but the second-order term is largest exactly there, and
this maximises the lateral speed that multiplies it. The margin and `certifiedDuration`
are what absorb inter-sample error, and the escape systematically pushes into the regime
where that error is biggest.

Both arms are non-zero before the escape exists -- the row is enforced at every sampled
state, while the audit uses a thinner barrier and finer sampling -- so this degrades an
already imperfect margin rather than introducing a new class of violation. It is still
the wrong direction for a safety filter, which is why the default stays zero.

## Capping the speed: the gain and the damage are the same thing

`escapeSpeed` scales the escape's aim as a share of the nominal's speed, so it controls
the lateral sweep directly -- the error the margin has to absorb between samples grows as
its square. Same 420 MotionBenchMaker problems, all four arms solving all of them:

| escape speed | median ms | paired time | paired samples | unsafe/audited | rate | worst clearance |
| --- | --- | --- | --- | --- | --- | --- |
| off | 0.911 | 1.000x | 1.000x | 1541/293665 | 0.525% | -0.0122 m |
| 0.25 | 0.960 | 1.040x | 1.002x | 1542/293723 | 0.525% | -0.0122 m |
| 0.50 | 0.981 | 1.058x | 0.994x | 1517/292967 | 0.518% | -0.0122 m |
| 1.00 | 0.935 | 1.001x | 0.950x | 1670/288460 | 0.579% | -0.0213 m |

Halving the speed restores the audit exactly -- 0.518% against 0.525%, and a deepest
incursion identical to the off arm's to four decimals -- and removes the whole sample
gain with it. There is no knee. **The 5% of samples and the doubled incursion were the
same phenomenon**: the escape bought motion by spending margin, and capping the speed
stops it doing either. At 0.50 it is worse than useless, since every escape call still
pays a second QP: same samples, audit restored, 5.8% slower.

A certificate-based cap was tried first and is vacuous. Requiring the escape's
`safeDuration` to cover the step suppressed it outright -- `durations()` bounds travel by
the worst lever arm over every sphere, and the rollout it feeds already runs ordinary
controls whose safe span is under one step, since `roll()` floors the span at `stepSize`.
Holding the escape to a standard the rest of the system does not apply left the cost of
the second QP and none of the motion: on the mode-2 probe it reproduced the escape-off
numbers to three digits at +22% wall time.

## Verdict

`escapeProgress` stays at zero. What it buys is margin, not planning skill, and there is
no setting that keeps the samples and returns the margin.

That sharpens the case for the window rather than weakening it. A pointwise rule can only
trade margin for motion, because sweeping sideways at speed is the only thing it can do
with no information about where sideways leads. A window objective picks a direction that
goes somewhere, so its lateral motion is paid for in progress instead. Whether it also
erodes the margin is not settled by anything here -- it would sweep near surfaces too --
but it has something to show for it, and this does not.

`esc_s<seed>_e<0.00|0.25>.csv` are the off and full-speed arms, `esc_s<seed>_v<0.50|0.25>.csv`
the capped ones.
