#!/usr/bin/env bash
# All 689 eligible, with the forearm/wrist_2 pair margin scaled below its own floor.
#
# Established tonight: sphere pair 14-21 (forearm_link vs wrist_2_link) has a clearance
# FLOOR of +20.4 mm -- those spheres never touch at any configuration -- and carries a
# 44.71 mm calibrated margin, 2.19x that floor. It is the only one of the 52 margined
# pairs with a positive floor; the other 51 genuinely overlap somewhere, so their margins
# legitimately let the row bind. This one turns a pair that cannot collide into a row that
# is violated across most of its range. It accounts for 120 of the 136 screened endpoints
# and all 33 problems that admitted no path, and at k<0.456 the row goes vacuous and all
# 33 solve in a median 6 ms.
#
# k=0.25 is a DIAGNOSTIC, not a proposed value. 44.71 mm is a sampled supremum over
# configurations where the two links' CONVEX HULLS overlapped, and the forearm's hull is
# 1.55x its mesh -- so the right fix is re-calibrating that pair against triangle meshes,
# not turning this knob down. These runs measure what the pair margin is costing; they do
# not propose a replacement.
#
#   D1  max eligibility, zero reserve. Exact field (buffer 0), margin 0, self buffer 0.
#       Answers "can all 689 be eligible and reachable at once". Expect ~688-689 eligible
#       -- one endpoint binds on pair 13-22, whose floor is -46.9 mm, so scaling cannot
#       make it vacuous and should not. Safety will be bad by construction: the same
#       configuration without the pair fix gave 42 unsafe rows at -16.5 mm.
#
#   D2  the one that matters. Grid + derived 3 mm buffer -- the only configuration all
#       evening that stayed audit-clean -- with the pair margin scaled. Eligibility is
#       capped at ~669 by the 3 mm world screen, which is the honest price of safety.
#       This measures how much of the 153 comes back WITHOUT giving up the guarantee.
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_pairfix_all689_20260914
ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope

export OMPL_CBF_SCREENING=1
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.25
export OMPL_MBM_ROWS=$ROWS

run() {  # tag shards selfbuffer extra_env
    local tag=$1 shards=$2 selfbuf=$3 extra=${4:-}
    local per=$(( $(nproc) / shards ))
    for i in $(seq 0 $((shards - 1))); do
        OMPL_MBM_SELF_BUFFER=$selfbuf OMPL_MBM_SHARD=$i/$shards \
          taskset -c "$(( i * per ))-$(( i * per + per - 1 ))" \
          env $extra "$BIN" "$OUT/scenes_all.txt" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
            "$OUT/${tag}_shard$i.csv" 1 0 4 0 0 10000 \
            >"$OUT/${tag}_shard$i.log" 2>"$OUT/${tag}_shard$i.err" &
    done
    wait
    head -1 "$OUT/${tag}_shard0.csv" >"$OUT/$tag.csv"
    for i in $(seq 0 $((shards - 1))); do tail -n +2 "$OUT/${tag}_shard$i.csv" >>"$OUT/$tag.csv"; done
    echo "$tag rows: $(tail -n +2 "$OUT/$tag.csv" | wc -l)"
}

# D1: exact field -> no bake, ~36 MB a shard, so 24 shards is fine.
run d1_maxeligible 24 0 "OMPL_MBM_EXACT_FIELD=1"
# D2: grid -> 3.7 GB a shard, so 7 shards as in every grid run tonight.
run d2_safe_3mm_buffer 7 0.001
