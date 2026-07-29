# Picard iteration starter

This directory contains a scalar, allocation-free Picard trajectory iteration.
It intentionally has no SIMD, threading, CBF, robot, or QP dependency.

## Hot-path properties

- Compile-time state dimension and maximum node count.
- No heap allocation during construction or solve.
- Two fixed trajectory buffers; iterations swap buffer indices rather than copy trajectories.
- Templated vector-field callback, so the CBF control callback can inline.
- All vector-field evaluations in one round read the previous trajectory only.
- Fixed iteration cap and finite-value checks.

## CBF integration shape

For a six-joint robot:

```cpp
using Picard = ompl::picard::PicardIteration<6, 32>;
Picard solver(options);

solver.solveLinear(start, goal,
    [&](const Picard::State &q, double t, Picard::State &dqdt)
    {
        // 1. Form the nominal goal-seeking velocity at q and t.
        // 2. Evaluate the SDF/barrier rows at q.
        // 3. Run the CBF projection/QP.
        // 4. Store the filtered velocity in dqdt.
    });
```

The solver does not certify safety. After the final Picard round, perform the
batched node and segment safety checks before accepting the trajectory.
