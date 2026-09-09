# Hold-time bounds: derivations and measured value

Review of the spatial hold-time module (`hold_time.py`, from the *Verify Tighten Bounds*
conversation) against the certificates in `ClearanceBarrier` / `CertifiedRegionRollout`,
plus the normal-projected variant derived during the review.

Everything below was checked numerically against exact forward kinematics. Scripts are
listed in [Reproducing](#reproducing).

---

## 1. The problem

A point `p` on the chain follows the straight joint-space ray

```
q(t) = q0 + t·v            v constant
```

Given an initial free-ball radius `d0` around `p(0)`, the **hold time** `τ` is a certified
lower bound on the first time `p` can have left that ball:

```
‖p(t) − p(0)‖ ≤ d0        for all t ∈ [0, τ]
```

With `v` a unit vector, `t` *is* joint-space arc length, so `τ` is directly the certified
step length that `CertifiedRegionRollout` wants from `safeScale()`.

### Notation

| symbol | meaning |
|---|---|
| `z_j` | world unit axis of joint `j` at `q0`; attached to link `j−1` |
| `s_j` | segment `o_{j+1} − o_j`, with `o_n = p`; rotates with link `j` |
| `l_j` | `‖s_j‖` (rigid, motion-invariant) |
| `ω_j` | angular velocity of link `j` = `Σ_{k≤j} v_k z_k(t)` |
| `α_j` | `ω̇_j` |
| `χ_k` | bound on `‖ż_k‖` |
| `J₀` | position Jacobian at `q0`, column `k` = `z_k × Σ_{j≥k} s_j` |
| `Ω_j, A_j, W_j` | horizon envelopes for `‖ω_j‖`, `‖α_j‖`, `‖α̇_j‖` |
| `C, V, H` | envelopes for `‖p̈‖`, `‖ṗ‖`, `‖p⃛‖` |

---

## 2. What we ship today: the L1 lever-arm bound

`ClearanceBarrier::travelBound()` is `leverArms() * |Δq|`, certifying the polytope

```
Σ_k L_ik · |Δq_k|  ≤  slack_i
```

Restricted to the ray `Δq = s·u` this gives

```
s_repo = min_i  slack_i / (L_i · |u|)
```

`L_ik` bounds `|∂p_i/∂q_k| = dist(p_i, axis_k)` **over the whole configuration space**
(`UR5::leverArmBounds()`, swept axial-ball and origin-ball enclosures). That global maximum
is the looseness: `CertifiedRegionRollout.h:56` notes the resulting 19.8 mm self-collision
slack is *"loose by however far the arm is from its worst pose."*

Note `UR5::armLengthBounds()` — the naive downstream reach sum — is only *"retained as the
ablation baseline and an upper cap for leverArmBounds()"* (`UR5.h:310`). **This matters:
`hold_time.weighted_l1` with `reach_radii` is exactly `armLengthBounds()`, so the module's
advertised ~1.8× is measured against the bound we already discarded.**

---

## 3. The hold-time construction

### 3.1 Speed-capped displacement

With `S = ‖ṗ(0)‖ = ‖J₀v‖` and valid horizon envelopes `C ≥ ‖p̈‖`, `V ≥ ‖ṗ‖` on `[0,T]`:

```
‖p(t) − p(0)‖  ≤  ∫₀ᵗ min(S + C·s, V) ds
```

Setting that equal to `d0` and inverting gives `speed_capped_time`. Clipping the root at `T`
is valid: if the root exceeds `T` the bound never exhausted `d0` inside the horizon.

### 3.2 Point envelopes from link envelopes

```
ṗ = Σ_j ω_j × s_j                              ⇒  V = Σ_j l_j·Ω_j
p̈ = Σ_j [ α_j × s_j + ω_j × (ω_j × s_j) ]      ⇒  C = Σ_j l_j·(A_j + Ω_j²)
```

Differentiating `p̈` once more, with `r_j` the world segment:

```
r⃛_j = α̇_j × r_j + 2 α_j × (ω_j × r_j) + ω_j × (α_j × r_j) + ω_j × (ω_j × (ω_j × r_j))
⇒  H = Σ_j l_j·(W_j + 3·A_j·Ω_j + Ω_j³)
```

which is the module's jerk expression. Verified analytically — it holds with margin
everywhere.

### 3.3 The cosine-interval span recursion

`Ω_j` needs `‖Σ_{k≤j} v_k z_k(t)‖` over the horizon, so it needs bounds on the pairwise
cosines `c_km = z_k·z_m` as the axes move. The key observation:

> Rotating about joint `k` moves `z_m` rigidly **around** `z_k`, so it cannot change the
> angle between them. Likewise joint `m` rotates about `z_m` itself.

So `angle(z_k, z_m)` is driven **only by the interior joints `k+1 .. m−1`**, and

```
|d/dt angle(z_k, z_m)|  ≤  ‖Σ_{r=k+1}^{m−1} v_r z_r‖  =  Ω[k+1, m−1]
```

Widen `θ⁰_km` by `T·Ω[k+1,m−1]`, clamp to `[0,π]`, take cosine endpoints. Adjacent axes
(`m = k+1`) have **constant** cosine — no interior joints. Spans are then accumulated by
inclusion–exclusion, in increasing gap so the interior span is always already known:

```
Q[k,m] = Q[k+1,m] + Q[k,m−1] − Q[k+1,m−1] + 2·v_k·v_m·c_selected[k,m]
```

with `c_selected` the interval endpoint maximising the signed product. `Ω_j = √Q[0,j]`.
Each pair is maximised independently, so the selected values need not form a PSD Gram
matrix — that makes the box bound conservative, not invalid.

`χ_k` refines `‖ż_k‖ = ‖ω_{k−1} × z_k‖` by removing the axial component:
`χ_k = √(Q[0,k−1] − m²)` where `m` lower-bounds `|ω_{k−1}·z_k|` from the same intervals.
Then `A_j = Σ_{k≤j} |v_k|·χ_k` and `W_j = Σ_{k≤j} |v_k|·(A_{k−1} + Ω_{k−1}·χ_k)`.

### 3.4 Cost structure (the important practical fact)

**The `O(n²)` span recursion — `Ω`, `χ`, `A`, `W`, the cosine intervals — depends only on
the joint axes and `v`, not on the point.** Verified: two different segment sets sharing the
same axes give bit-identical arrays. Only `C`, `V`, `H` (three dot products with `l`) and
`S = ‖J₀v‖` are per-point.

So a hop pays the recursion **once** and each of the 40 spheres costs `O(n)`.

---

## 4. Self-collision reduces to a shorter chain

This is the constraint that binds our planner, and it needs no second kinematic chain. Our
own derivation at `UR5.h:829` already supplies the structure. For spheres `a` on frame `f`
and `b` on frame `g`, `f ≤ g`:

- `k < f` — joint `k` rotates both spheres **as one rigid body**:
  `J_a,k − J_b,k = z_k × (p_a − p_b)`, and `n·(z_k × n)‖p_a − p_b‖ = 0`. Identically zero.
- `f ≤ k < g` — only `b` moves.
- `k ≥ g` — neither moves.

Take frame `f` as the base. Sphere `a` is rigid in it, so `b` is a **single material point**
on the sub-chain driven by joints `f..g−1`:

```
axes     = z_f .. z_{g−1}                    (expressed in frame f)
segments = o_{f+1}−o_f, …, p_b − o_{g−1}
d0       = ‖p_a − p_b‖ − r_a − r_b − margin
```

Median **2 active joints of 6** across the 303-pair table — strictly easier than the full
arm.

### Why the isotropic bound does poorly here

All four delivered variants certify

```
‖p_a − p_b‖(t)  ≥  ‖p_a − p_b‖(0) − δ(t),      δ(t) = ‖p_b(t) − p_b(0)‖
```

which is **direction-agnostic**: it spends the whole clearance budget as though every
millimetre of motion were straight down the line between the spheres. Real self-pair motion
is mostly tangential — the wrist swings *past* the forearm. Measured radial share of speed:
median `−0.000`, and the pair is *receding* 50% of the time.

---

## 5. The normal-projected bound

Work in frame `f`, `w(t) = p_a − p_b(t)`, and freeze `n = w(0)/‖w(0)‖` at `t=0`:

```
‖w(t)‖  ≥  n · w(t)                     (Cauchy–Schwarz, any fixed unit n)
        =  ‖w(0)‖ − n · δ(t)            (δ(t) = p_b(t) − p_b(0), vector)

⇒  c(t)  ≥  d0 − n · δ(t)
```

The budget is spent by the **signed radial component** `n·δ`, not by `‖δ‖`.

Two properties make this cheap:

1. **Freezing `n` is sound, not approximate.** Cauchy–Schwarz holds for *any* fixed unit
   vector, so there is no normal-rotation term to track and no extra horizon restriction. As
   the pair swings, the bound goes loose — never invalid.
2. **A sphere is convex**, so the step is tight to first order. For a general world SDF the
   same move needs the obstacle inside a separating halfspace (true for convex obstacles) or
   a curvature/reach bound — which is why this is the *self-pair* fix specifically.

### 5.1 Naive version — does not work

Substituting into the existing scalar machinery:

```
n·δ(t) = ∫₀ᵗ n·ṗ_b(s) ds
σ₀ = n·(J₀v)          signed; |σ₀| ≤ ‖J₀v‖
n·p̈_b ≤ C,  n·ṗ_b ≤ V

n·δ(t) ≤ ∫₀ᵗ min(σ₀ + C·s, V) ds
```

i.e. the one-line substitution `S = ‖J₀v‖ → σ₀`. This needs a **signed** scalar solver,
since `speed_capped_time` requires `S ≥ 0`:

```
t = (−σ₀ + √(σ₀² + 2·C·d0)) / C          for σ₀ of either sign, below the V cap
```

**Measured: median 0.99× — no gain.** With `σ₀ ≈ 0` the bound is dominated by the crude
`C·t²/2` envelope term, and it loses to `experimental_anchored`, which uses the *exact*
initial acceleration rather than an envelope.

### 5.2 Projecting the anchored bound — this is the one

Keep both the exact initial motion and the normal:

```
n·δ(s) = s·(n·u₀) + s²·(n·a₀)/2 + R(s),        |R(s)| ≤ H·s³/6

τ = first t with   max   [ s·(n·u₀) + s²·(n·a₀)/2 ]  +  H·t³/6  =  d0
                  s ≤ t
```

`u₀ = J₀v` and `a₀ = p̈(0)` exactly (`Geometry.initial_motion`); both `n·u₀` and `n·a₀` are
signed and small when the pair slides past. The **prefix maximum** is required — the raw
anchored expression is non-monotone, and a late endpoint that has returned inside must not
hide an earlier crossing. The prefix max of a scalar quadratic is just its endpoints plus,
when concave, the vertex `s = −b/c`.

Because it is not uniformly better (min 0.78×), take the max over independently valid
certificates.

---

## 6. Verification

Exact FK simulator (rotations about reference axes composed in order) plus analytic
derivatives, checking every envelope against ground truth.

| quantity | worst excess over ~400 randomized chains |
|---|---|
| `Ω_j`, `χ_k`, `A_j`, `W_j` | ≤ 2.6e-13 (attained, i.e. valid and tight at `t=0`) |
| `C`, `V`, `H` | negative — bound never reached |
| cosine intervals | ≤ 2.8e-15 |
| displacement vs `speed_capped_bound` | 0.0 |
| `τ` vs exact first-exit | 0.0 |

Regimes: generic chains (n=3..9), fast rates over long horizons, axes near-parallel to 1e-3,
horizons down to 1e-4. Plus 60 UR5 world-obstacle configurations and 3347 self-pair
evaluations across 150 configurations. **No violation anywhere.**

The module also reproduces its four documented seconds exactly:
`0.084520, 0.151953, 0.152717, 0.177837`.

> A jerk-envelope violation I reported mid-review was my own third-derivative finite
> difference (roundoff ≈ 0.1 at `h = 1e-5`). Computed analytically, `H` holds with margin.

---

## 7. Measured tightness

Fraction of the exact first-exit time; `1.000` = a certificate equal to truth, `> 1.000`
would be unsound. Global lever arms are **sampled** maxima, a lower bound on the real
`leverArmBounds()` table, so every gain below is a floor.

### World obstacles — UR5, 60 configurations, horizon 0.5 s

| bound | fraction of exact exit |
|---|---|
| `weighted_l1` (= `armLengthBounds`) — the module's own baseline | 0.289 |
| L1, exact global lever arms — best case for what we ship | 0.360 |
| L1, exact *local* lever arms — not a certificate, reference only | 0.701 |
| `level2` | 0.950 |
| `tightened_level2`, 2 passes | 0.952 |
| `experimental_anchored` | **0.998** |

Pairwise `level2` / exact-global-lever-arm L1: **median 2.45×**.

### Self-pairs — UR5, 150 configurations, 281 exiting samples of 3347, horizon 1.5 s

| bound | fraction of exact exit | vs. best isotropic |
|---|---|---|
| L1, exact global lever arms | 0.356 | — |
| `level2` | 0.564 | — |
| `experimental_anchored` | 0.682 | baseline |
| normal, level-2 envelopes (§5.1) | 0.696 | 0.99× |
| **normal, anchored (§5.2)** | **0.813** | **1.14×** |
| max of all certificates | 0.815 | 1.14× (min 1.00×) |

Normal-anchored pairwise ratios: median 1.14×, mean 1.31×, p90 1.65×, min 0.78×.

---

## 8. What it is worth in practice

Per-pair tightness ratios are not the answer, because the planner takes the **min over all
303 self-pairs** — what matters is the gain on the *binding* constraint. So this walks a
straight edge exactly the way `CertifiedRegionRollout::plan` does (1.5 rad edges,
`minAdvance = 0.01`, `maxEvaluations = 40`, `shrink = 1 − 1e-9`), with all 40 spheres and all
303 self-pairs, comparing:

- **repo** — `s = min_p slack_p / (L_p · |u|)`, the L1 lever-arm polytope on the ray
- **hold** — max over the independently valid hold-time certificates per pair

Self-collision limited, i.e. the open-space regime: with the world 8 m away the header
reports 12.7 m of world slack against 19.8 mm of self slack, so self-pairs are what bind.

| | repo | hold | |
|---|---|---|---|
| first-hop certified radius (median) | 0.2304 rad | 0.7666 rad | **3.01×** |
| evaluations per edge (mean) | 8.66 | 2.92 | |
| evaluations per edge (median ratio) | — | — | **0.38× — 62% fewer** |
| rad per evaluation (median ratio) | — | — | **2.67×** |
| travel per edge | 1.236 rad | 1.249 rad | 1.00× |
| outcomes | 26 complete, 12 stalled | 26 complete, 12 stalled | identical |

38 edges. The two methods reach the **same place** with the **same terminal outcomes** — the
tighter certificate simply gets there in 62% fewer barrier evaluations. The header's quoted
0.12 rad certified radius in open space is consistent with the `repo` column here.

### Measured C++ cost

Benchmarked against the real `ClearanceBarrier` and `UR5` (4000 random `(q,u)`, `-O3
-march=native`, obstacle parked out of reach so self-collision binds). Certificate set is
`max(level2, normal-anchored, L1)` — the same set the walk above used, minus the isotropic
anchored bound, which never binds on self-pairs.

| stage | µs/hop |
|---|---|
| `certifiedRegion(q)` — FK + 40 centres + SDF batch + 343 slacks | 1.228 |
| `travelBound` + `safeScale` — 343-row ray restriction | 0.294 |
| **current hop** | **1.523** |
| repo L1, 303 self-pairs | 0.357 |
| envelope build alone — one `O(n²)` pass serving all pairs | 0.306 |
| hold-time self rows, 14.7 pairs evaluated/hop | 1.843 |
| **hop with hold-time self rows** | **3.071** |

```
cost ratio      2.02x        break-even 2.67x
net             1.31x        (median-of-ratios)
per edge        8.66 x 1.523 = 13.2 us   ->   2.92 x 3.071 = 9.0 us   =  1.47x
```

**It wins, by roughly 1.3–1.5× in wall time**, but the margin is thinner than the 2.67×
evaluation saving suggests. Three things matter:

1. **The isotropic anchored bound has to go.** It costs 2.2 µs/hop on its own (4.06 vs 1.84)
   and contributes nothing on self-pairs — `hold` and `hold+iso` return bit-identical scales
   on every config checked, consistent with 0.813 vs 0.815 in §7. Keeping it would put the
   cost ratio at 4.4×, well past break-even, and the whole thing would lose.
2. **The measurement is pessimistic.** `(a) + hold` computes the 303 pair clearances twice —
   once inside `certifiedRegion`, once inside the hold path. A fused implementation saves
   roughly 0.3 µs, moving the ratio to about 1.8× and the net past 1.45×.
3. **Headroom remains.** 14.7 pairs are evaluated per hop; the screen seeds from the single
   smallest L1 bound and then prunes. Processing in ascending L1 order would cut the
   dominant per-pair cost further.

Soundness was re-verified in C++ against exact FK: worst `certified − true first contact` =
`−2.81e-04` over 400 configurations. Bisection uses 12 iterations — 3.7e-4 rad on a 1.5 rad
horizon, far below `minAdvance = 0.01`.

---

## 9. Recommendations

| | action | why |
|---|---|---|
| **Adopt** | `level2` + `experimental_anchored` on world-clearance rows | 2.45× longer certified hops, 0.998 of exact. Replaces `safeScale()` on the ray, not `certifiedRegion()`'s polytope — and the rollout already committed to a ray. |
| **Adopt** | max(normal-anchored, existing L1) on self-pairs | 1.14× median, 1.65× p90, never worse. Sub-chain is 2 joints; `n` is already computed in `selfPairClearance()`. |
| **Drop** | `tightened_level2` | 0.950 → 0.952 for a second envelope evaluation. |
| **Needed** | signed scalar solver | `speed_capped_time` requires `S ≥ 0`; the projected rate is signed and half of all self-pairs are receding. |

---

## 10. Limits

- The end-to-end walk (§8) uses all 40 spheres and all 303 self-pairs; the tightness tables
  in §7 use 21 representative spheres and no gripper spheres.
- The C++ benchmark covers the self-collision rows only; a hold-time path for world rows is
  neither implemented nor timed, and in clutter (where world rows bind) the sliding rollout
  is the better tool anyway.
- The benchmark is a micro-benchmark of the certify call, not a planner run: no RRT, no
  ledger, no `FilteredMotionValidator`.
- UR5 only — Baxter's longer chains were not exercised.
- `d0` sampled uniformly rather than drawn from the real SDF field; the end-to-end walk is
  self-collision limited (the open-space regime), with no world obstacles.
- Global lever arms sampled over 1200–1500 configurations, so they under-estimate
  `leverArmBounds()` and all gains are conservative floors.
- Float64 throughout, not outward-rounded interval arithmetic. Adequate at planning
  tolerances; not a formal proof, as the module itself states.
- C++ per-hop cost is estimated from the operation count, not measured.
- Nothing here enforces the CBF envelope `dh/dt ≥ −κh`; like `CertifiedRegionRollout`, it
  gives that up.

---

## Reproducing

Scripts in the session scratchpad under `ht/`:

| script | what it does |
|---|---|
| `hold_time.py` | the module under review, as delivered |
| `verify.py`, `verify2.py` | randomized envelope + soundness checks against exact FK |
| `ur5.py` | world-obstacle comparison on repo UR5 geometry |
| `normal_aware.py` | §5.1, signed scalar solver |
| `normal_anchored.py` | §5.2, the recommended variant |
| `fast_bench2.py` | self-pair tightness table |
| `ur5_tables.py` | 40 spheres + 303 self-pairs parsed from `UR5.h` |
| `practice.py` | end-to-end walk mirroring `CertifiedRegionRollout::plan` |
