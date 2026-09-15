#!/usr/bin/env bash
# Is the world penetration caused by the per-step sphere-centre travel?
#
# qpFixed is the FIXED-step row: 0.01126 rad/call, essentially one integration step, and
# it still penetrates -10.199 mm. One step cannot produce 10 mm of penetration unless the
# step itself sweeps that far:
#
#     OMPL_CBF_MAX_SPEED 10 rad/s  x  stepSize 0.001 s = 0.01 rad per step
#     0.01 rad  x  lever arm up to ~1 m               = up to 10 mm of centre travel
#
# The filter certifies h >= 0 at the step ENDPOINTS; the audit densifies and checks the
# INTERIOR. A centre translating 10 mm past a shelf corner is clear at both ends and
# inside in the middle. Consistent with the rest: depth tracks hop length (qpEnvelope
# longest at 0.0387 rad/call and deepest at -16.5 mm), and both gate rows -- which take
# no certified step -- sit at exactly zero.
#
# If that is the mechanism, penetration falls with max speed and costs NO eligibility,
# where the 3 mm buffer costs 20 problems. If it does not fall, the mechanism is
# something else and the buffer is the only lever we have.
#
# 60 problems: every one that any QP row penetrated in D1. Running the population where
# the effect exists, not all 689 -- this is a mechanism test under time pressure.
#
# Confound to keep in mind when reading SOLVE counts: segFrac -1 ties RRT-Connect's edge
# resolution to maxSpeed x stepSize, so lowering the speed also refines the baseline's
# edge checking. That is fine for the penetration question and not for solve-rate claims.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-24}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_maxspeed_sweep_20260914

export OMPL_MBM_EXACT_FIELD=1
export OMPL_CBF_SCREENING=1
export OMPL_CBF_JOINT_LIMITS=0
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.25
export OMPL_MBM_SELF_BUFFER=0
export OMPL_MBM_ROWS=qpFixed,qpAdaptive,qpEnvelope,qpFreeEnvelope

for v in 10 5 2 1; do
    for i in $(seq 0 $((SHARDS - 1))); do
        OMPL_CBF_MAX_SPEED=$v OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
            "$BIN" "$OUT/scenes_violators.txt" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
            "$OUT/v${v}_shard$i.csv" 1 0 4 0 0 10000 \
            >"$OUT/v${v}_shard$i.log" 2>"$OUT/v${v}_shard$i.err" &
    done
    wait
    head -1 "$OUT/v${v}_shard0.csv" >"$OUT/v$v.csv"
    for i in $(seq 0 $((SHARDS - 1))); do tail -n +2 "$OUT/v${v}_shard$i.csv" >>"$OUT/v$v.csv"; done
    echo "maxspeed=$v rows: $(tail -n +2 "$OUT/v$v.csv" | wc -l)"
done
