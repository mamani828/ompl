#!/usr/bin/env bash
# D1 with and without the post-solve landing check (OMPL_CBF_VERIFY_STEP).
#
# D1 = exact analytic field, margin 0, buffer 0, self buffer 0, pair margin k=0.25.
# It is the configuration that gets all 689 eligible and reachable, and the one with the
# worst penetration: 52-53 of 687 returned paths contain a real environment collision,
# median 5-6% of the states in a bad path and up to a third, worst -16.5 mm. Not one is a
# self-collision. The two accept-or-stop rows are at exactly zero across 564852 states.
#
# The fix under test: after the QP solves and the step is integrated, evaluate the
# enforced barrier at the landing state. If it is negative the linear model that certified
# the step was wrong, so refuse it and keep the previous state -- already verified, being
# either the rollout's start or a landing that passed the same test. The planner gets a
# shorter rollout and decides. One barrier evaluation per committed step, against the ~343
# rows the filter already assembles per call.
#
# off/on are the same binary, same seed, same everything, one env var apart.
#
# What would falsify the fix: violations that survive because a hop spans more than one
# step (qpAdaptive and qpEnvelope run a certified control up to 0.0387 rad, and the audit
# samples at 0.02 rad, so an interior point can dip while both endpoints pass). qpFixed
# hops 0.0113 rad -- under the audit spacing -- so if anything is still dirty THERE, the
# mechanism is not what this fix assumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
SHARDS=24
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_verifystep_d1_20260915
export OMPL_MBM_EXACT_FIELD=1 OMPL_CBF_SCREENING=1 OMPL_CBF_MAX_SPEED=10 OMPL_CBF_JOINT_LIMITS=0
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.25 OMPL_MBM_SELF_BUFFER=0
export OMPL_MBM_ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope
for mode in off on; do
    [ $mode = on ] && V=1 || V=0
    for i in $(seq 0 $((SHARDS-1))); do
        OMPL_CBF_VERIFY_STEP=$V OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
          "$BIN" "$OUT/scenes_all.txt" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
          "$OUT/verify_${mode}_shard$i.csv" 1 0 4 0 0 10000 \
          >"$OUT/verify_${mode}_shard$i.log" 2>"$OUT/verify_${mode}_shard$i.err" &
    done
    wait
    head -1 "$OUT/verify_${mode}_shard0.csv" >"$OUT/verify_$mode.csv"
    for i in $(seq 0 $((SHARDS-1))); do tail -n +2 "$OUT/verify_${mode}_shard$i.csv" >>"$OUT/verify_$mode.csv"; done
    echo "verify=$mode rows: $(tail -n +2 "$OUT/verify_$mode.csv" | wc -l)"
done
