#!/usr/bin/env bash
# Does the forearm/wrist_2 pair margin close the corridor, or does the filter?
#
# The 33 problems here are eligible at the sound config and yet admit no path from any
# row -- isSafe (plain RRT-Connect, no CBF), qpFixed, qpAdaptive, qpEnvelope alike -- not
# at 10 s and not at 120 s, where they burn ~12 M samples for zero solutions. VAMP solves
# all 33 in a median 0.13 ms on the same 40 spheres and the same 383 self pairs. So the
# free space is open and something in this configuration closes it.
#
# VAMP differs from our rows in two ways at once: no calibrated pair margin, and a
# discrete checker rather than a CBF filter. This sweep separates them. Scaling
# `UR5::selfPairMargins()` moves only the first. If the 33 start solving as k falls, the
# margin is what closed the corridor; if they stay blocked at k=0 -- where our self rows
# become exactly VAMP's zero-margin sphere test -- the margin is exonerated and the
# filter's certified step is the remaining suspect.
#
# VAMP runs as a control row at every k. It ignores our margins entirely, so its 33/33
# must not move; if it does, the harness is not doing what this script claims.
#
# k=1.0 is the shipped calibration and must reproduce the 0/33 already measured.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-24}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_selfmargin_sweep_33_20260914

export OMPL_MBM_EXACT_FIELD=1
export OMPL_CBF_SCREENING=1
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0
export OMPL_MBM_ROWS=isSafe,qpFixed,qpAdaptive,qpEnvelope,VAMP

for k in 1.0 0.75 0.5 0.25 0.0; do
    for i in $(seq 0 $((SHARDS - 1))); do
        OMPL_UR5_SELF_PAIR_MARGIN_SCALE=$k OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
            "$BIN" "$OUT/scenes_33.txt" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
            "$OUT/k${k}_shard$i.csv" 1 0 4 0 0 10000 \
            >"$OUT/k${k}_shard$i.log" 2>"$OUT/k${k}_shard$i.err" &
    done
    wait
    head -1 "$OUT/k${k}_shard0.csv" >"$OUT/k$k.csv"
    for i in $(seq 0 $((SHARDS - 1))); do tail -n +2 "$OUT/k${k}_shard$i.csv" >>"$OUT/k$k.csv"; done
    echo "k=$k rows: $(tail -n +2 "$OUT/k$k.csv" | wc -l)"
done
