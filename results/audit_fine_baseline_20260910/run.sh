#!/usr/bin/env bash
# Does the baseline's collision-check spacing survive an audit finer than the margin?
#
# `isSafe` only -- segmentFraction reaches no other row (UR5MBMBenchmark.cpp:1377), so
# this is the whole experiment. Two configs, each covering all 689 problems:
#   matched  segFrac -1        -> maxSpeed*stepSize = 0.0100 rad
#   fine     segFrac 0.000325  -> 0.0050 rad, what every earlier run used
# Audited at 0.002 rad, below the 3 mm buffer under test, instead of the 0.02 rad
# default that is an order of magnitude coarser than the margin it reports on.
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/audit_fine_baseline_20260910
export OMPL_CBF_SCREENING=1 OMPL_MBM_ROWS=isSafe OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0 OMPL_MBM_AUDIT_RES=0.002 OMPL_SDF_BAKE_THREADS=3

core=0
for i in 0 1 2 3; do
    OMPL_MBM_SHARD=$i/4 taskset -c "$core-$((core+2))" "$BIN" "$OUT/scenes_all.txt" \
        100 10.0 0.003 0.001 2.0 0.0 0.003 -1 20 -1 0.0 "" -1 1 \
        "$OUT/matched$i.csv" 1 0 4 0 0 10000 >"$OUT/matched$i.log" 2>"$OUT/matched$i.err" &
    core=$((core+3))
done
for i in 0 1 2; do
    OMPL_MBM_SHARD=$i/3 taskset -c "$core-$((core+2))" "$BIN" "$OUT/scenes_all.txt" \
        100 10.0 0.003 0.001 2.0 0.0 0.003 0.000325 20 -1 0.0 "" -1 1 \
        "$OUT/fine$i.csv" 1 0 4 0 0 10000 >"$OUT/fine$i.log" 2>"$OUT/fine$i.err" &
    core=$((core+3))
done
wait
for tag in matched fine; do
    first=1
    for f in "$OUT/$tag"*.csv; do
        if [ $first -eq 1 ]; then head -1 "$f" >"$OUT/$tag.csv"; first=0; fi
        tail -n +2 "$f" >>"$OUT/$tag.csv"
    done
    echo "$tag: $(tail -n +2 "$OUT/$tag.csv" | wc -l) rows"
done
