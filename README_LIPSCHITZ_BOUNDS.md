# UR5 swept-enclosure Lipschitz bounds

`UR5::leverArmBounds()` now bounds sphere motion using a search over swept
cylinders, balls centered on the current joint axis, and balls centered at the
joint origin. The construction follows the geometry in the local
`safe_bubble_cover/src/safe_bubble_cover/envs/lipschitz_3d.py` implementation.
It uses the UR5 model's existing joint transforms and sphere centers directly;
no Python package, generated sphere ordering, or external repository is needed
at runtime.

Each enclosure contains the whole distal rotational sweep. All three branches
are retained at each upstream joint because a locally tighter primitive can
be worse upstream. Each table entry takes the minimum over sound enclosures
and the original arm-length bound. The cylinder projection includes the
axial/rim cross term needed for tilted axes. Enclosures receive a small outward
floating-point padding. These are analytic geometric bounds with numerical
padding, not formally verified interval arithmetic.

The table is computed once per process. Runtime matrix dimensions and the
screening/certificate algorithms are unchanged. Existing relative-motion masks
apply the tighter coefficients to retained self-collision pairs too. Pair
selection, collision margins, and QP settings are unchanged. This implementation
only changes UR5; its rotational construction is not a general solution for
robots with prismatic joints.

The original table remains available as `UR5::armLengthBounds()`. For an ablation,
set `OMPL_UR5_LEVER_BOUNDS=arm_length` before starting the process. The default
uses swept bounds. Changing this variable after bounds have been initialized
does not change cached tables.

## Reproduction

```bash
cmake --build build --target demo_UR5MBMBenchmark test_ur5 test_clearance_barrier test_cbf_control_filter test_filtered_state_space -j 4
ctest --test-dir build -R '^(test_ur5|test_clearance_barrier|test_cbf_control_filter|test_filtered_state_space)$' --output-on-failure
python3 scripts/benchmark_lipschitz_bounds.py
```

The runner compares the same binary with both bound tables, alternates order
across seeds 17, 29, and 43, and saves raw CSVs and logs. It selects ten problems
per scene, allows one second per planner query, enables self-collision with
zero additional self-margin (the model's per-pair margins remain), and uses
safe hops, kappa 8/s, 0.05 s integration steps, and a 0.02 m SDF grid.
Audits are the existing benchmark's sampled audits, not continuous ground-truth
mesh validation. A matched RNG seed does not imply identical trees once hop
lengths change.

## Validation

The coefficient sum falls from 110.256 to 87.3351 (20.79%). No entry increases.
Tests check sampled analytic sphere Jacobians, self-pair gradients over 3,000
configurations, and cylinder/ball enclosure containment with tilted axes.
These numerical checks supplement the enclosure derivation; they do not prove
that the underlying sphere model or sampled pair pruning covers mesh collision.

The screening-equivalence test permits roundoff in the reported gain and also
checks agreement of the screened and full QP controls. Different active-set
paths can produce one-ULP differences even for equivalent constraints.

All four targeted CTest suites passed (UR5, clearance barrier, CBF filter, and
filtered state space); the UR5 suite contains 13 test cases.

## Measured effects (2026-09-07)

Raw results are in `results/lipschitz_bounds/`. Of the 70 selected problems per
seed, 55 were eligible; three seeds give 165 eligible queries per mode. Both
modes solved the same 153 queries.

| Metric | Arm length | Swept enclosures |
| --- | ---: | ---: |
| Solved / eligible | 153 / 165 | 153 / 165 |
| Median solved-query time | 4.1297 ms | 4.1284 ms |
| Median filter calls, solved queries | 392 | 375 |
| Median joint travel per filter call | 0.03655 rad | 0.03802 rad |
| Total planning seconds, including failures | 14.7267 | 14.8941 |
| Sampled audited states | 107,221 | 106,802 |
| Environment-unsafe audit states | 20 | 15 |
| Paths with environment-unsafe audit states | 8 | 7 |
| Self-collision audit states | 0 | 0 |

For the 153 matched solved queries, the median of baseline/tight time ratios
is 1.036x and the median filter-call ratio is 1.030x. This is different from
the ratio of pooled median times. Total time includes twelve one-second
timeouts per mode. Worst audited environment clearance is -0.00685555 m in
both modes; these are not fully audit-clean runs.

Median paired time ratios by scene (greater than one favors tighter bounds):

| Scene | Baseline / tight |
| --- | ---: |
| bookshelf_small | 0.977x |
| bookshelf_tall | 1.218x |
| bookshelf_thin | 1.170x |
| box | 1.015x |
| cage | 0.902x |
| table_pick | 1.105x |
| table_under_pick | 1.027x |

The geometric improvement produces modest call-count savings, but runtime
effects are mixed. These three-seed measurements do not establish statistical
significance or a general planning-speed advantage. In particular, a 20.79%
reduction in coefficient sum is not a 20.79% speedup: the binding constraints,
minimum integration step, and changed RRT trajectories determine the outcome.
