# CBF-QP versus QP-free baselines

This branch contains controlled C++ comparisons between OMPL's CBF-QP with
Lipschitz rollout certificates and the accept-or-stop, no-QP CBF gate adapted
from LQR-CBF-RRT*.

The adaptation is intentional: the released LQR-CBF-RRT* implementation is a
Python, two-dimensional, custom-RRT* planner and has no UR5 model. Here only its
local safety decision is ported: apply the nominal control when every active CBF
inequality is satisfied and stop the rollout otherwise. Both rows use the same
robot, barriers, samples, bounds, planner, and termination budget.

## Build

Configure OMPL with demos and qpmad enabled, then build the two targets:

```bash
cmake -S . -B build -DOMPL_BUILD_DEMOS=ON
cmake --build build --target demo_PointCBFGateBenchmark demo_UR5QPGateBenchmark -j4
```

## Point robot, plain RRT

```bash
./build/demos/demo_PointCBFGateBenchmark 10000 5000 10017 > point_gate.csv
python3 scripts/summarize_point_cbf_gate.py point_gate.csv
```

The arguments are paired trials, RRT iteration budget, and first seed.

## UR5, RRT or RRTConnect

Generate `scenes.txt` as documented in `demos/UR5MBMBenchmark.cpp`, then run:

```bash
./build/demos/demo_UR5QPGateBenchmark scenes.txt \
  3 5000 0.03 0.05 2.0 0.005 0.03 19.8 217 ur5_rrt.csv rrt

./build/demos/demo_UR5QPGateBenchmark scenes.txt \
  3 5000 0.03 0.05 2.0 0.005 0.03 19.8 217 ur5_connect.csv rrtconnect

python3 scripts/summarize_ur5_cbf_gate.py ur5_connect.csv
```

The positional arguments are `scenes`, problems per scene, iterations, SDF voxel,
integration step, planner range, audited margin, filter buffer, CBF gain, first
seed, CSV path, and planner. Optional trailing arguments restrict the scene and
give a comma-separated problem-index allowlist.

The UR5 benchmark audits every returned executed path against exact MBM primitive
distances and the independent UR5 self-collision model. Do not use timing results
unless the world-collision, self-collision, and replay-miss totals are all zero.

## All-method MotionBenchMaker comparison

The existing `demo_UR5MBMBenchmark` target now runs these rows together:

- `rrtconnect`: explicit SDF collision checking;
- `qp-fixed`: our CBF-QP at the fixed integration step, with Lipschitz duration
  computation disabled;
- `qp-lipsch`: our CBF-QP using its Lipschitz rollout certificates;
- `qp-free`: the accept-or-stop gate, with neither QPs nor collision checking;
- `vamp-rrtc`: VAMP collision checking, when VAMP is available.

Every row receives an independently allocated sampler with the same per-problem seed.
The CBF rows use `AllValidStateValidityChecker`; their filter/rollout is the only
in-planner safety mechanism. The exact primitive and self-collision audit remains common
to all returned paths.

```bash
./build/demos/demo_UR5MBMBenchmark scenes.txt \
  10 5.0 0.03 0.05 2.0 0.005 0.03 -1 19.8 -1 0 \
  '' -1 17 results/mbm_all_methods.csv
```

`qp-fixed` still uses sound row screening because that cannot change the QP solution;
what it disables is the separate duration certificate and the coarse rollout hops that
spend it. It is therefore the fixed-step, no-Lipschitz-certificate A/B for `qp-lipsch`.
