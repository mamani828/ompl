# UR5 MotionBenchMaker at kappa = 20: envelope hold certificate against L1

MotionBenchMaker UR5, 140 problems (20 per scene, seven scenes), seed 1.
30 mm voxel, 4 mm audited margin plus a 5 mm filter buffer, 10 ms controller
step, 1.0 rad RRT-Connect range, 10 s limit, kappa 20 /s, uncapped certified
step, 0 mm self-collision margin over 303 pairs, unlimited per-extension
rollout budget, no rope shortcutting. The 30 mm field and the guard admit 80 of
the 140 problems; the other 60 have endpoint clearance below the guard and are
skipped identically for every row.

`holdNew` is excluded throughout -- it differs from `qpEnvelope` only in which
hold certificate answers the `safe` span. The row selection governs the summary
table and the CSV as well as the work, so a dropped row is absent from both
rather than appearing as a line of zeros indistinguishable from a row that ran
and solved nothing:

    OMPL_MBM_ROWS=isSafe,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,VAMP

### Three deliberate choices, all defaults of this benchmark

**QP row screening off.** Screening is sound and a pure speed win, but it makes a
filter call about half as expensive, and a hold certificate's cost is fixed per
call while its benefit is proportional to what a call costs. Benchmarking
certificates against a screened filter measures them in their least favourable
regime. `OMPL_CBF_SCREENING=1` restores it. **This is the wrong configuration for
a headline against RRT-Connect** -- see the caveat under Result.

**Arm-length L1 baseline.** `armLengthBounds()` bounds distance-to-axis by full
distance-to-origin, discarding the axial component; `leverArmBounds()` starts
there and tightens it with a swept-enclosure search. The tightening is this
repository's own improvement, so folding it into the baseline compares the
envelope against a stronger L1 than anything published. `qpEnvelope` scores
identically either way (1,415 against 1,413), so this moves `qpAdaptive` alone:
1,482 tightened, 1,523 arm-length. Pass `OMPL_UR5_LEVER_BOUNDS=lever_arm` for the
tightened baseline.

**QP shortcuts left on.** `Parameters::plainSolve` strips them -- the feasibility
bypass, the closed-form one-row projection, the pre-inverted Cholesky factor --
and it was measured rather than assumed: the two bypasses skip only **2.8%** of
calls (97.2% already reach the solver), the Cholesky factor is worth 7.6% of a
solve, and stripping all three costs **+2.2% per call**. The QP was never
meaningfully assisted, so the shortcuts stay on. `OMPL_CBF_PLAIN_QP=1` to check.

## Result

`final_armlength.csv` / `.log`.

| row | solved | median ms | evaluations | vertices | path | unsafe / audited |
|---|---:|---:|---:|---:|---:|---:|
| rrtconnect | 80/80 | 6.03 | 7,010 | 37 | 12.34 | 110 / 56,234 |
| qp-fixed | 80/80 | 8.15 | 2,030 | 15 | 8.31 | 432 / 147,596 |
| qp-adapt | 80/80 | 6.97 | 1,523 | 15 | 8.31 | 431 / 115,958 |
| qp-free | 80/80 | 8.43 | 8,050 | 66 | 16.03 | **0 / 176,657** |
| qp-env | 80/80 | 6.86 | **1,415** | 15 | 8.31 | 430 / 112,622 |
| vamp-rrtc | 80/80 | 0.10 | 1,958 | 31 | 12.26 | 308 / 52,653 |

    reduction  7.09%   (1523 -> 1415)
    overhead   0.9%    (0.045 us/call on a 4.878 us wrapped-QP call)
    net        +6.19%  predicted
    wall       6.86 vs 6.97 = -1.6%, inside the noise band

**Read the 7.09% with its baseline attached.** The envelope did not improve --
1,415 against 1,413 before this configuration. The baseline weakened, 1,482 to
1,523, because it now uses the published arm-length bound. Quoted against the
tightened lever-arm table the same envelope wins 4.66%.

### Caveats

**With screening off, both CBF rows are slower than plain RRT-Connect** (6.86 and
6.97 against 6.03). With screening on, `qp-adapt` runs 3.09 against 6.13 -- a 2x
win. The two questions need different configurations: screening off to compare
certificates with each other, screening on to compare the filter against a
sampling baseline. Do not quote this table against RRT-Connect.

**Only `qp-free` is audit-clean.** Zero unsafe audited states against ~430 for
both fast rows, worst clearance -2.2 mm. By the benchmark's own criterion --
*"non-zero unsafe invalidates a row however fast it was"* -- the fast rows are
invalid at a 4 mm margin and `qp-free` earns its 1.2x. That is a property of the
buffer, unchanged by any certificate work here; see the margin and voxel sweep.

**Wall time cannot resolve these differences.** `qp-adapt`'s own median varies
3.09-4.03 ms across identical runs, so a 6% effect is at the edge and a 2% one is
inside it. Barrier evaluations are deterministic and reproduce exactly. Report
the evaluation count with the measured per-call overhead; a wall-time claim needs
multiple seeds.

## The L1 ladder

All three bound the same quantity, `|dp_i/dq_k|`, and differ only in how much
geometry they keep. Fractions of true first exit are from
`README_HOLD_TIME_BOUNDS.md`:

| bound | what it discards | fraction of exact exit |
|---|---|---:|
| `armLengthBounds()` | the axial component, **and** maximises over q | 0.289 |
| `leverArmBounds()` | maximises over q | 0.360 |
| `rho_ik(q0)`, telescoping | nothing; exact at the current configuration | 0.701 |

The third rung was labelled *"not a certificate, reference only"*. It is now one --
see the telescoping term below.

## The telescoping local-lever term

For world sphere i, `rho_ik(q0) = dist(p_i(q0), axis_k(q0))` at the *current*
configuration certifies the whole straight ray `q0 + t u`:

    |p_i(q0 + t u) - p_i(q0)|  <=  t * sum_k rho_ik(q0) |u_k|

Reach `q0 + t u` one joint at a time, **proximal to distal**. At step k joints
1..k-1 have moved and joints k..n have not, so the earlier joints' change is a
*rigid* transform of the whole distal assembly -- joint k's axis and sphere i
together, still in their original relative pose. Rigid transforms preserve
distance, so the radius the sphere swings on at step k is exactly `rho_ik(q0)`,
and that step contributes `2 rho sin(|t u_k|/2) <= rho_ik(q0)|t u_k|`. Sum with
the triangle inequality. The bound is linear in t, hence monotone, so it also
bounds the supremum over `[0, t]` -- which a hold certificate needs and a
non-monotone bound would not give. Ordering is load-bearing: distal-first moves
joints k+1..f(i) before step k, changing the sphere's distance to axis k.

**Validated: 9,600,000 samples against exact forward kinematics, zero
violations**, worst residual -4.1e-17 (roundoff), over six horizons from 0.01 to
2.0 and all 40 spheres. And `rho_ik(q0) <= leverArmBounds()(i,k)` by
construction, so it is never weaker than the global term beside it.

Measured payoff: **none.** It settles 53.6% of expensive world rows before any
envelope is built -- inside the 30-50% band that would make it a good screen --
and changes nothing: evaluations 1,415 either way, certificate cost 0.045 against
0.045 us/call. It prunes the *cheap* half. `endpoint()` is ~2 square roots plus
warm cache reads (all 40 world spheres share frame 0, so `ensureFrame(0)` is warm
after the first row), while `localWorldRate` costs 6. The expensive parts --
`anchorEndpoint` and the root solve -- were already behind the existing
`t >= best` check, and the local term only reaches past it in the 1.6% of world
bindings where it beats the speed-capped term.

It is kept because it is cost-neutral and strictly tightens the bound, not
because it is faster. The chord refinement `2 rho sin(theta/2)` is dead on
arrival here: at 0.5 rad/s and 50 ms hops `theta ~ 0.025 rad`, and the validation
run measured worst `actual/bound = 1.000000` -- the arc bound is already tight to
six decimals, with a relative gap of `theta^2/24 ~ 2.6e-5`.

The same argument extends to self pairs through a pair's common ancestor frame
and is not implemented: self rows bind a minority of the time and their own L1
already wins 77% of those.

## The certificate's worth scales with the cost of a filter call

    net%  =  evaluation_reduction%  -  certificate_overhead%

The reduction is proportional to whatever a filter call costs; the certificate's
cost is a fixed ~0.035 us per call. So the net rises as calls get dearer, toward
the full reduction. Swept over a 2.6x range of per-call cost, with the arm-length
L1 table and everything else held fixed:

| regime | us/call | qp-adapt | qp-env | reduction | overhead | net |
|---|---:|---:|---:|---:|---:|---:|
| screening + `activePairs` | 1.869 | 1,525 | 1,415 | 7.21% | 1.9% | **+5.31%** |
| screening only | 2.283 | 1,523 | 1,415 | 7.09% | 1.5% | **+5.59%** |
| no screening (default here) | 4.878 | 1,523 | 1,415 | 7.09% | 0.9% | **+6.19%** |

The reduction is flat at 7.1-7.2% and only the overhead moves. Nothing else
varies: same L1 table, same certificate, `qp-env` at 1,415 in all three. The
model is confirmed at the other end too -- with the *tightened* lever-arm table
the same sweep gives 1.5% overhead screened against 1.1% unscreened for an
identical 4.66% reduction.

**So the result, stated once:** against the published arm-length L1 certificate,
the envelope removes **7.1% of barrier evaluations** at **0.9-1.9% overhead**
depending on the cost of a filter call, for a net **+5.3% to +6.2%**.

`activePairs` is not a pure speed knob -- it caps the certificate at
`pairRelevance * max(dt, 1/kappa)` = 0.2 s. Against arm-length L1 that cap barely
bites (`qp-adapt` 1,523 to 1,525); against the tightened table it does
(1,482 to 1,495), so a sweep using it must say which L1 it ran. It is sound with
the envelope: dropped pairs are proven clear for `relevance * scale` while
`EnvelopeHoldFilter` clips at `1 * scale`, which is conservative, and `qp-env`
scores 1,413 either way.

### This benchmark is the certificate's worst regime

UR5 with 40 spheres, 303 pairs and active screening is about as cheap as a filter
call gets, and the certificates are fixed-cost. Anywhere calls are dearer -- more
collision geometry, a finer field, a higher-DOF arm -- the same certificates are
worth more with no code change. **The call-reduction figures are therefore the
robust, hardware-independent result**: 7.1% for the envelope over arm-length L1,
and for the adaptive hold over a fixed step, 1,523 against 2,030 evaluations
(25%), which reads -2.5% in wall time on a cheap call and -16.3% on a dear one.

### And it wins its real fight in the fast regime

    qp-adapt   2.60 ms      rrtconnect  6.03 ms      2.3x
    qp-env     2.64 ms      vamp-rrtc   0.10 ms
    qp-fixed   2.67 ms

Screening and `activePairs` on is both the fastest and the shipping
configuration. The three CBF rows land within 3% of each other in wall time,
which is why the evaluation counts carry the comparison.

## Margin and voxel sweep

The remaining axis, swept because it acts on both the certificate's ceiling and
the audit failures. `Barrier::interpolationBuffer(field)` is `field.spacing()`
-- the voxel size -- so the *sound* buffer at a 30 mm grid is 30 mm, and every
run above passes 5 mm, six times less.

Full set, 20 per scene, 20 ms step, 4 mm margin, 5 mm buffer:

| | voxel 30 mm | voxel 10 mm |
|---|---:|---:|
| problems admitted | 80/140 | 50/140 |
| qp-adapt evaluations | 770 | 1,184 |
| qp-env evaluations | 738 (-4.16%) | 1,158 (-2.20%) |
| qp-adapt unsafe / audited | 215 / 67,256 | 15 / 48,844 |
| qp-env unsafe / audited | 214 / 66,195 | 21 / 47,785 |
| worst clearance | -0.0021 | -0.0038 |
| qp-free unsafe / audited | 0 / 85,164 | 0 / 56,956 |

**A finer voxel does not fix the audit; it measures it more honestly.** Unsafe
states fall roughly tenfold but not to zero, and worst clearance gets *worse*
(-2.1 mm to -3.8 mm) because the coarse field was smoothing real penetration
into something shallower than it is. The same effect explains the admitted count
falling from 80 to 90 skipped: a 30 mm field over-reports clearance near
surfaces, so endpoints pass a guard they should not.

On problems solved by every arm (7 of 35, 5 per scene), the buffer is what
matters and margin is not:

| voxel / buffer / margin | adapt ev | env ev | reduction | env unsafe |
|---|---:|---:|---:|---:|
| 30 / 5 / 4 mm | 27,446 | 27,038 | 1.49% | 40 / 5,202 |
| 15 / 15 / 4 mm | 68,378 | 67,281 | 1.60% | 0 / 6,453 |
| 10 / 10 / 4 mm | 45,665 | 44,787 | 1.92% | 0 / 5,283 |
| 10 / 5 / 4 mm | 27,231 | 26,812 | 1.54% | 0 / 5,237 |
| 10 / 10 / 0 mm | 27,333 | 26,907 | 1.56% | 0 / 5,277 |

Dropping the margin to zero changes nothing (27,333 against 27,231), so margin
is not the lever. The sound buffer costs +68% evaluations at a 10 mm voxel
(45,665 against 27,231) and is the only setting that reached zero unsafe on
every subset tested.

**The certificate's ceiling does not move with geometry.** The reduction is
1.5-1.9% on the harder common set and 4-5.5% on the easier one, tracking problem
mix rather than slack. Combined with the step-size sweep (4.85% at 50 ms falling
monotonically to 2.31% at 2 ms), the envelope's advantage over L1 is a robust
but small 2-5% across every operating point tested. Nothing in this sweep
changes that conclusion.

## Files

| file | what |
|---|---|
| `final_armlength.csv` / `.log` | **the run above**: current defaults -- screening off, arm-length L1, optimized QP, escalation on, gate ceiling 2, local-lever term on |
| `final_screened.csv` / `.log` | same with `OMPL_CBF_SCREENING=1`, full row set -- the **shipping** configuration: 2.263 us/call, 1,523 against 1,415, net +5.59%, 1.87x over RRT-Connect |
| `final_fast_activepairs.csv` / `.log` | fastest regime: screening + `OMPL_CBF_ACTIVE_PAIRS=1`; 1.869 us/call, net +5.31%, and the 2.3x over RRT-Connect |
| `final_plainqp1.csv` / `.log` | same with `OMPL_CBF_PLAIN_QP=1`; the +2.2%-per-call unassisted solve |
| `final_plainqp0.csv` / `.log` | its paired control, shortcuts on, tightened lever-arm L1 |
| `seed1_range1_final.csv` / `.log` | earlier canonical run: screening **on**, tightened L1 (1,482 against 1,413) |
| `seed1_range1_final_timed.csv` / `.log` | same, `OMPL_HOLD_TIMING=1`; source of the screened cost model |
| `seed1_range1_no_holdnew.csv` / `.log` | pre-escalation baseline (`unbound` 9.6%, 1,449 evaluations) |
| `seed1_range1_escalate.csv` / `.log` | escalation only, before the ceiling cap (1,413 evaluations, 2.9% overhead) |
| `seed1_range1_step005_nogate.csv` / `.log` | 50 ms step, gate off; where `qp-free` being the only clean row is clearest |
| `seed1.csv`, `seed1_budget200.csv`, `seed1_range1_budget200.csv`, `seed1_range1_unlimited_stall_escape.csv` | predate the row selection and the instrumentation; all eight rows including `holdNew` |

CSVs carry three added columns: `hop_floored`, `hop_at_region`,
`hop_edge_limited`.

## Environment switches

| variable | default | effect |
|---|---|---|
| `OMPL_MBM_ROWS` | all | comma-separated row selection; governs work, table and CSV |
| `OMPL_CBF_SCREENING` | 0 (off) | QP row screening; 1 halves the per-call cost and is the shipping choice |
| `OMPL_UR5_LEVER_BOUNDS` | `arm_length` | `lever_arm` for the tightened swept-enclosure table |
| `OMPL_CBF_PLAIN_QP` | 0 | 1 strips the feasibility bypass, one-row projection and Cholesky factor |
| `OMPL_CBF_SCREEN_SCALE` | 1.0 | widens the screening horizon; 16 costs +56% for nothing |
| `OMPL_ENV_ESCALATE` | 1 | re-ask unscreened when the answer rests on the `1/kappa` clip |
| `OMPL_ENV_CEILING` | 2.0 | cap on the gate's adaptive gain ceiling |
| `OMPL_ENV_GATE` | 1 | 0 runs a query on every call: +0.09pp conversion, overhead 1.4% -> 12.6% |
| `OMPL_ENV_SUBSET` | 1 | 0 always sweeps all 343 rows |
| `OMPL_HOLD_TIMING` | off | per-call certificate cost breakdown |
| `OMPL_CBF_PROFILE` | off | per-component wall time |
