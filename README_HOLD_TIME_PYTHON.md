# Certified hold times in Python

[`hold_time.py`](hold_time.py) computes conservative hold times for material
points on a fixed-base spatial serial revolute chain moving along

```text
q(t) = q0 + t v
```

The module supports point-to-obstacle clearance and self-collision between
point-centred covering balls. It uses NumPy and caches the geometry at `q0`.

## Requirements and tests

Python 3.10 or newer and NumPy are required.

```bash
python3 -m unittest -v test_hold_time
python3 hold_time.py
```

The unit tests include dense forward-kinematics trajectories. These are
regression checks; the certificate follows from the analytic bounds implemented
by the module.

## Geometry convention

For a point with `n` upstream revolute joints, construct `Geometry` with:

- `axes[j]`: the world-space unit axis of joint `j` at `q0`.
- `segments[j]`: `o[j+1] - o[j]`, with the last endpoint replaced by the
  material point. Segment `j` rotates with link `j`.
- `v[j]`: the constant rate of joint `j`, in radians per second.

Lengths, clearances, and ball radii must use the same unit. Times are seconds.

```python
import numpy as np
from hold_time import Geometry, level2, tightened_level2

g = Geometry(
    axes=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
    segments=[[.5, 0, .1], [.1, .4, 0], [.2, .1, .3]],
)
v = np.array([.8, -.7, .5])

# The point starts with 0.15 length units of free clearance. Bounds are valid
# only on the supplied 1.0 second envelope horizon.
result = level2(g, v, d0=.15, horizon=1.0)
print(result.tau)

# Spend up to four envelope evaluations tightening the same certificate.
tight = tightened_level2(g, v, d0=.15, max_horizon=1.0, passes=4)
print(tight.tau, tight.evaluations)
```

`HoldResult.tau` is always clipped to the horizon whose envelopes justify it.
`root` may exceed that horizon and must not be used directly.

## Point-to-obstacle clearance

For one material point, `d0` is a valid initial free-ball radius. To cover a
rigid robot link, use enough point-centred balls to cover its entire geometry:

1. Query clearance at each ball centre.
2. Subtract that ball's radius and any desired numerical margin.
3. Compute a hold for every centre.
4. Take the minimum hold.

A single closest witness at `q0` does not cover a link over a finite motion.

The available point certificates are:

| Function | Purpose |
|---|---|
| `weighted_l1` | Global downstream-reach baseline. |
| `level2` | Recommended one-pass speed-capped certificate. |
| `tightened_level2` | Re-evaluates envelopes on shorter horizons. |
| `experimental_anchored` | Second-order point bound with a jerk remainder. |

`experimental_anchored` returns the larger of its anchored hold and Level-2,
so it cannot reduce the Level-2 result.

## Self-collision

For two covering-ball centres `a` and `b`, define

```text
d0 = ||point_a - point_b|| - radius_a - radius_b - margin
```

Every enabled non-adjacent ball pair must be represented. The self-collision
certificate does not become a full robot certificate when applied only to the
pair that is closest at `q0`.

`frame_a` and `frame_b` are counts of upstream joints. They lie in `[0, n]`.
The helper swaps the points when needed and cancels joints upstream of both
frames. It returns a shorter relative chain containing only the joints that
move one point relative to the other.

```python
import numpy as np
from hold_time import (
    SelfCollisionPair,
    minimum_self_collision_hold,
    relative_self_collision_geometry,
    self_collision_hold,
)

axes = np.array([
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.],
])
joint_origins = np.array([
    [0., 0., 0.],
    [.4, 0., 0.],
    [.7, .2, 0.],
])
point_a = np.array([.1, .25, 0.])   # attached after joint 0
point_b = np.array([.9, .25, .2])   # attached after joint 2

geometry, normal = relative_self_collision_geometry(
    axes, joint_origins, point_a, point_b,
    frame_a=1, frame_b=3,
)
relative_rates = np.array([1.1, -.8])  # full_v[1:3]
d0 = np.linalg.norm(point_a - point_b) - .15 - .20

pair_hold = self_collision_hold(
    geometry, relative_rates, d0, horizon=1.2, normal=normal,
)

pairs = [
    SelfCollisionPair(geometry, relative_rates, d0, normal),
    # Add every other enabled covering-ball pair here.
]
robot_hold = minimum_self_collision_hold(pairs, horizon=1.2)
```

For each pair, `self_collision_hold` returns

```text
min(horizon, max(weighted L1, Level-2, fixed-normal anchored))
```

The maximum is valid because each term independently certifies the same pair.
The minimum across pairs is required because every pair must remain separated.
If the existing system has a tighter independently certified old-L1 pair hold,
pass it as `l1_time`; the function will preserve it in the maximum.

When both points share a frame, the helper returns `geometry=None`. Their
relative placement is rigid and their hold equals the horizon when `d0 > 0`.

## Assumptions

The analytic bounds require all of the following:

- A fixed-base serial chain with revolute joints.
- Constant joint rates over the returned hold.
- Static obstacles.
- Unit joint axes and finite input data.
- Valid initial clearances and a complete covering-ball model.
- Envelope data rebuilt whenever `q0` changes.

Boundary contact is allowed by the implementation's `<=` convention. Subtract
a positive clearance margin when the application requires strict separation or
needs to absorb modelling and floating-point error.

The implementation uses float64 arithmetic. It is an analytic certificate
under exact arithmetic, but float64 is not an outward-rounded interval proof.
Safety-critical numerical certification requires validated arithmetic and
conservatively rounded geometry, rates, and clearances.

## API summary

| API | Return value |
|---|---|
| `Geometry(axes, segments)` | Cached point-chain geometry at `q0`. |
| `Geometry.envelopes(v, T)` | Spatial speed, acceleration, and jerk envelopes on `[0,T]`. |
| `weighted_l1(d0, v, reach_radii)` | Baseline hold, possibly infinity. |
| `level2(...)` | One-pass `HoldResult`. |
| `tightened_level2(...)` | Budgeted multi-pass `HoldResult`. |
| `experimental_anchored(...)` | Anchored-plus-Level-2 `HoldResult`. |
| `relative_self_collision_geometry(...)` | Relative `Geometry` and initial unit normal. |
| `normal_anchored_time(...)` | Fixed-normal pair hold as a float. |
| `self_collision_hold(...)` | Best certified hold for one pair. |
| `minimum_self_collision_hold(...)` | Limiting hold over all supplied pairs. |

The full derivation and benchmark discussion are in
[`README_HOLD_TIME_BOUNDS.md`](README_HOLD_TIME_BOUNDS.md).
