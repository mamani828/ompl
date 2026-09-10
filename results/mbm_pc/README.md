# MotionBenchMaker UR5 -- probabilistic-completeness A/B

MotionBenchMaker UR5, 140 problems (20 per scene, 7 scenes), 5 s limit, kappa = 8 /s,
seeds 1-3. Two binaries differing in exactly one line of `FilteredStateSpace::roll()`:
whether the tiny-control early-termination test is gated on the budget remaining *after*
the step. Ungated, it refused every extension shorter than
`minControlFraction * stepSize * maxSpeed` = 2.5e-6 rad -- including ones that were
arriving -- which is a positive lower bound on extension length, and the ball-covering
argument of the completeness proof cannot have one. See `pc_proof.tex`,
Lemma 2 and the remark following it.

- `prePC_instr_s<seed>.csv` -- ungated build (the violation)
- `postPC_s<seed>.csv`      -- gated build (current `main`)
- `postPC_prof.csv`/`.log` -- one profiled run, `OMPL_CBF_PROFILE=1`, seed 1.
  Timings there are NOT comparable: `FilterStats::record` takes a process-wide mutex
  on every filter call. It is for the filter-outcome and certificate columns only.

## Command

```
demo_UR5MBMBenchmark scenes20.txt 20 5.0 0.03 0.05 2.0 0.004 0.005 -1 8.0 -1 0.0 "" -1 <seed> <csv>
#                    scenes  per  s   vox  step rng marg buf  sf  kap  mss sm  -   -   seed  out
```

## Provenance

- git: 57586e39 on cbf-self-collision
- pre-fix binary md5:  a039b4b1a95f305039adb4a21ba8172d
- post-fix binary md5: 92a52d14199fe3085b157143ac9fae33
- generated: 2026-09-03T22:53:19Z

## Headline

Tiny-control terminations over the three seeds: 105 of 391001 rollouts pre-fix, 5
post-fix. Each of the 100 eliminated was a rollout that would have arrived. Solve rate
140/140 in both arms at every seed; medians within 3%.

Plots: `./scripts/plot_pc_ab.py results/mbm_pc`
