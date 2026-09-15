# All-689 fair accept-or-stop benchmark

This section compares eight planners on all 689 MotionBenchMaker UR5 problems.
The two no-QP rows use the vanilla accept-or-stop rule: the nominal control is
applied unchanged only when it satisfies every active CBF row; otherwise the
extension stops. `OMPL_QPFREE_REPAIR=0` explicitly disables scaling/braking.

## Fairness controls

- Seed 1, with the same locally seeded, absolute-ordinal-derived sample stream for
  every OMPL method on a given problem. Sharding does not change that stream;
  RRTConnect's otherwise constructed internal RNG is unused.
- The same 0.1 rad goal tolerance, 2.0 rad planner range, 10 s time limit, and
  eligibility test for every method.
- The ordinary RRTConnect validity-check spacing is matched to the filtered
  rollout step.
- Exact analytic scene fields, zero world/self filter buffer, and the same
  post-solution audit for all methods.
- Twenty-four process-isolated shards, pinned one per logical CPU. All methods
  for a problem execute within its assigned shard.

The complete environment, source diff, commit, and binary hash are retained in
this directory. `run.sh` reproduces and combines the shards.

## Results

| Method | Eligible | Solved | Audited environment-unsafe states | Audited self-colliding states |
|---|---:|---:|---:|---:|
| `isSafe` | 689 | 689 | 1 | 0 |
| `qpPlain` | 689 | 689 | 0 | 0 |
| `qpFixed` | 689 | 689 | 0 | 0 |
| `qpAdaptive` | 689 | 689 | 0 | 0 |
| `qpFreeGate` | 689 | 601 | 0 | 0 |
| `qpEnvelope` | 689 | 689 | 0 | 0 |
| `qpFreeEnvelope` | 689 | 631 | 0 | 0 |
| `VAMP` | 689 | 686 | 7 | 0 |

Both no-QP rows recorded zero repaired controls. Their only actions were to pass
the nominal control unchanged or refuse it. The envelope raises the vanilla
gate's solve count from 601 to 631 by certifying longer holds; it does not alter
control selection.

This is a matched single-seed comparison. It supports paired per-problem
comparisons, but uncertainty claims should use additional independent seeds.
