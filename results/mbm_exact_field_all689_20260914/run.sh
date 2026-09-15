#!/usr/bin/env bash
# UR5 MotionBenchMaker over all 689 problems with the scene's CLOSED-FORM distance field
# instead of a baked 3 mm grid (OMPL_MBM_EXACT_FIELD=1).
#
# Why: these scenes are 12-15 boxes and cylinders with exact signed distance, so the grid
# is a speed cache, not a model -- and it is a cache that costs accuracy exactly where
# this benchmark cannot afford it. `interpolationBuffer()` reserves one voxel (3 mm here)
# against the interpolant disagreeing with the function it sampled, while
# MotionBenchMaker's goals sit a median 8 mm off the shelf: measured over all 689
# endpoints, a 3 mm buffer puts 20 of them inside an obstacle, where 1 mm puts none.
#
# What this does NOT fix, established on a 12-problem pilot before launching: the buffer
# absorbs TWO errors, and exactness removes only one of them. A CBF step linearises
# h(q + u dt) through the robot's nonlinear kinematic map -- the sphere centres travel on
# arcs and the constraint row is a chord -- and that error is there whatever the field is.
# The pilot's exact+zero-buffer qpAdaptive row landed at -4.048 mm where grid+3 mm was
# clean. Exactness also raises the hop length, because an exact SDF is honestly
# 1-Lipschitz where the interpolant measures above 1, so `maxGradientNorm()` drops and
# the certified step grows.
#
# Hence two configurations, same seed, same shard layout, one variable between them:
#   A  buffer -1  -> derived, and derived is 0 for an exact field. Direct A/B against
#                    mbm_voxel3mm_step0001_qprows_all689_nomargins_20260914, which is
#                    grid + buffer forced to 0.
#   B  buffer 1 mm -> the configuration that can actually win: enough for linearisation
#                    alone, and it costs none of the 689 endpoints.
#
# Shard layout is copied from the grid run (7 shards, 3 pinned cores each) so the
# comparison is not confounded by scheduling. Exact mode needs no grid, so peak RSS is
# ~36 MB per shard rather than 3.7 GB -- memory is no longer what bounds SHARDS, but
# matching the grid run matters more than going wider.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-7}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_exact_field_all689_20260914
SCENES=$OUT/scenes_all.txt

export OMPL_MBM_EXACT_FIELD=1
export OMPL_CBF_SCREENING=1
export OMPL_MBM_IGNORE_ELIGIBILITY=1
export OMPL_MBM_SELF_BUFFER=0
export OMPL_MBM_ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0

PER=$(( $(nproc) / SHARDS ))

run_config() {
    local tag=$1 buffer=$2
    #          scenes perScene secs voxel  step range margin  buffer  segFrac kappa maxStep selfMargin prefix shortcut seed
    local ARGS=("$SCENES"  100  10.0 0.003 0.001   2.0    0.0 "$buffer"     -1    20      -1        0.0     ""       -1    1 )
    local TAIL=( 1 0 4 0 0 10000 )
    for i in $(seq 0 $((SHARDS - 1))); do
        local LO=$(( i * PER )) HI=$(( i * PER + PER - 1 ))
        OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$LO-$HI" \
            "$BIN" "${ARGS[@]}" "$OUT/${tag}_shard$i.csv" "${TAIL[@]}" \
            >"$OUT/${tag}_shard$i.log" 2>"$OUT/${tag}_shard$i.err" &
    done
    wait
    head -1 "$OUT/${tag}_shard0.csv" >"$OUT/$tag.csv"
    for i in $(seq 0 $((SHARDS - 1))); do
        tail -n +2 "$OUT/${tag}_shard$i.csv" >>"$OUT/$tag.csv"
    done
    echo "$tag rows: $(tail -n +2 "$OUT/$tag.csv" | wc -l)"
}

# Verify the hand-differentiated obstacle gradients against central differences before
# either configuration runs. An exact field whose gradient describes a different surface
# from its value would reintroduce precisely the violations this run exists to remove.
OMPL_MBM_EXACT_FIELD=2 OMPL_MBM_ROWS=isSafe OMPL_MBM_SHARD=0/689 \
    "$BIN" "$SCENES" 100 0.1 0.003 0.001 2.0 0.0 0.0 -1 20 -1 0.0 "" -1 1 \
    "$OUT/gradcheck.csv" 1 0 4 0 0 10000 2>&1 | head -1 | tee "$OUT/gradcheck.log"

run_config exact_buffer0 -1
run_config exact_buffer1mm 0.001
