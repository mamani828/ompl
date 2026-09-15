#!/usr/bin/env bash
# All 689 MotionBenchMaker UR5 problems, self-pair margins at k=0.25, landing verification on.
#
# Two independent defects were found and are both addressed here.
#
# 1. Self-pair 14-21 (forearm_link vs wrist_2_link) carries a 44.71 mm calibrated margin
#    against a clearance FLOOR of +20.4 mm -- those two spheres never touch at any
#    configuration, and it is the only one of 52 margined pairs with a positive floor.
#    It rejected 120 of 136 screened endpoints and closed the corridor on 33 problems
#    that VAMP solves in a median 0.13 ms.  Measured against the triangle meshes rather
#    than PyBullet's convex hulls the requirement is 25.00 mm, not 44.71 -- 44% of it was
#    hull inflation -- but 25.00 is still above the floor, so re-calibration alone does
#    not reopen the corridor.  The pair is KEPT; its margin is scaled.
#
#    k=0.25 puts it at 11.18 mm, below the floor, so the row cannot fire.  Evidence that
#    this is safe: over 32202 executed states on the 33 tightest problems, 5718 states sat
#    below the 25.00 mm requirement and NONE put the meshes into contact; the two goals
#    the margin still rejects have 8.24 mm and 5.76 mm of real mesh separation.  The other
#    12 forearm/wrist_2 pairs stay at margin 0 and continue to enforce sphere non-overlap.
#
# 2. The QP rows committed steps their linear model certified but never checked, putting a
#    real environment collision in 7.6-7.7% of returned paths, up to a third of a path's
#    states and -16.5 mm deep.  OMPL_CBF_VERIFY_STEP evaluates the enforced barrier at each
#    landing and keeps the previous state instead of committing a bad one.  Mode 2 skips
#    the landings a hold-time certificate already covers -- sound, and ~44% of the checks
#    on qpAdaptive/qpEnvelope.
#
# margin 0 and buffer 0: the exact analytic field has no interpolation error to reserve
# against, so `interpolationBuffer()` derives 0 and the eligibility screen costs nothing on
# the world side.  687 of 689 are eligible; the 2 exclusions are near-contact goals that
# bind on pair 13-22, whose floor is -46.9 mm -- that pair's margin is legitimately live
# and is NOT scaled away by this run's evidence.
#
# VAMP runs as a reference row: same 40 spheres, same 383 self pairs, zero margin.
set -uo pipefail
cd "$(dirname "$0")/../.."
SHARDS=24
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_final_all689_k025_20260915

export OMPL_MBM_EXACT_FIELD=1
export OMPL_CBF_VERIFY_STEP=2
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.25
export OMPL_MBM_SELF_BUFFER=0
export OMPL_CBF_SCREENING=1
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0
export OMPL_MBM_ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope,VAMP

for i in $(seq 0 $((SHARDS-1))); do
  OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
    "$BIN" "$OUT/scenes_all.txt" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
    "$OUT/shard$i.csv" 1 0 4 0 0 10000 >"$OUT/shard$i.log" 2>"$OUT/shard$i.err" &
done
wait
head -1 "$OUT/shard0.csv" >"$OUT/all689_k025.csv"
for i in $(seq 0 $((SHARDS-1))); do tail -n +2 "$OUT/shard$i.csv" >>"$OUT/all689_k025.csv"; done
echo "rows: $(tail -n +2 "$OUT/all689_k025.csv" | wc -l)"
