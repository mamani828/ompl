#!/usr/bin/env bash
# Two experiments, back to back, both on the exact closed-form field.
#
# STEP 1 -- are the 33 genuine timeouts?
#   The 33 problems that no method solves inside 10 s, at the sound config. They are the
#   same set for isSafe/qpPlain/qpFixed/qpAdaptive/qpEnvelope -- plain RRT-Connect with no
#   CBF fails them too -- and the set does not move when the controller step changes 10x.
#   That rules out a CBF artefact but not a clock artefact. This gives them 12x the time
#   (120 s) on the grid config that produced them. If most solve, they are hard-but-
#   solvable and belong in the denominator; if none do, they are structurally blocked and
#   the honest report is that the benchmark contains them.
#
# STEP 2 -- a real 3 mm margin (config C).
#   Every 3 mm run so far spent its reserve on `interpolationBuffer()`: margin 0, buffer
#   3 mm, so the audited guarantee was 0 mm of clearance and the 3 mm only bought back the
#   grid's own interpolation error. On the exact field that buffer is genuinely 0, so the
#   same 3 mm can go into the *margin* instead: same screen (3 mm total, ~552 eligible)
#   but the audit now asserts 3 mm of real clearance rather than none.
#
#   Known risk, measured beforehand: the exact field does not fix the CBF's chord-through-
#   an-arc linearisation error, which reached -16 mm below the enforced level at buffer 0.
#   Expect violations of the 3 mm audit. A violation there still means 0-3 mm clear, not a
#   collision -- read `min_clearance` against the margin, not against zero.
#
# Seven rows in one run, so qpFreeEnvelope comes from the same process as the rest rather
# than being concatenated from a second run afterwards.
#
# Shard layout matches the grid runs (7 shards, 3 pinned cores each) so neither step is
# confounded by scheduling. Exact mode needs no baked grid (~36 MB per shard, not 3.7 GB),
# so memory no longer bounds SHARDS -- matching the runs being compared against does.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-7}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_margin3mm_exact_20260914

export OMPL_CBF_SCREENING=1
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0
PER=$(( $(nproc) / SHARDS ))

# scenes perScene secs voxel step range margin buffer segFrac kappa maxStep selfMargin prefix shortcut seed
run_config() {
    local tag=$1 scenes=$2 secs=$3 margin=$4 rows=$5 extra=${6:-}
    local ARGS=("$scenes" 100 "$secs" 0.003 0.001 2.0 "$margin" -1 -1 20 -1 0.0 "" -1 1)
    local TAIL=( 1 0 4 0 0 10000 )
    for i in $(seq 0 $((SHARDS - 1))); do
        local LO=$(( i * PER )) HI=$(( i * PER + PER - 1 ))
        OMPL_MBM_ROWS="$rows" OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$LO-$HI" \
            env $extra "$BIN" "${ARGS[@]}" "$OUT/${tag}_shard$i.csv" "${TAIL[@]}" \
            >"$OUT/${tag}_shard$i.log" 2>"$OUT/${tag}_shard$i.err" &
    done
    wait
    head -1 "$OUT/${tag}_shard0.csv" >"$OUT/$tag.csv"
    for i in $(seq 0 $((SHARDS - 1))); do tail -n +2 "$OUT/${tag}_shard$i.csv" >>"$OUT/$tag.csv"; done
    echo "$tag rows: $(tail -n +2 "$OUT/$tag.csv" | wc -l)"
}

# Step 1: the 33, on the grid config that produced them, with 12x the clock.
run_config timeout33_120s "$OUT/scenes_33.txt" 120.0 0.0 "isSafe,qpAdaptive"

# Step 2: config C. Exact field, margin 3 mm, buffer derived (= 0), eligibility screen ON
# because the eligible count at a real 3 mm is the thing being measured.
run_config margin3mm_exact "$OUT/scenes_all.txt" 10.0 0.003 \
    "isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope" \
    "OMPL_MBM_EXACT_FIELD=1"
