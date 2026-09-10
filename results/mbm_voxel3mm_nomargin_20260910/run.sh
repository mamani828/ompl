#!/usr/bin/env bash
# UR5 MotionBenchMaker at a 3 mm voxel with no CBF margins.
#
# Five rows: RRT-Connect (isSafe), QP fixed step, QP L1 certificate (qpAdaptive),
# no-QP CBF gate (qpFreeGate), QP envelope certificate (qpEnvelope).
# Screening on, RRT-Connect range 2.0 rad, margin 0 and filter buffer 0.
#
# Sharded over the problem set: each problem keeps the seed it would have had in a
# single-process run, so `cat` of the shard CSVs is the unsharded CSV. One 3 mm field
# is ~3.7 GB, so SHARDS is bounded by memory, not cores.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-7}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_voxel3mm_nomargin_20260910
SCENES=$OUT/scenes_all.txt

export OMPL_CBF_SCREENING=1
export OMPL_MBM_ROWS=isSafe,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope
export OMPL_SDF_BAKE_THREADS=$(( $(nproc) / SHARDS ))
# Lift the Constant(0.5) placeholder cap so it is not the binding constraint. It cannot
# be zero: FilteredStateSpace rejects non-positive speeds and the QP needs a bounded
# control box. See effectiveMaxSpeed().
export OMPL_CBF_MAX_SPEED=10
# Off: the geometric space already bounds sampled configurations, so this isolates the
# effect of additionally constraining one Euler step inside the QP.
export OMPL_CBF_JOINT_LIMITS=0

# segFrac is pinned rather than derived. It defaults to rolloutStep/extent, which with
# maxSpeed 10 would put the baseline's collision checks 0.1 rad apart -- far too coarse
# for 20 mm shelf panels, and it would flatter the baseline. 0.000325 holds it at the
# 0.0050 rad every earlier run used.
#          scenes perScene secs voxel step range margin buffer  segFrac kappa maxStep selfMargin prefix shortcut seed
ARGS=("$SCENES"      100  10.0 0.003  0.01   2.0    0.0  0.003 0.000325    20      -1        0.0     ""       -1    1 )
#  csvPath is appended per shard below, then: safeHops picardIter picardWindow picardWorkers trajPrefixes rolloutBudget
TAIL=( 1 0 4 0 0 1000 )

# Pin each shard to its own cores. Planning is single-threaded, so a pinned shard gets
# a dedicated core for it and the shards cannot steal each other's throughput. That
# matters for rows that hit the time limit: their sample count is bounded by wall-clock
# progress rather than by the seed, so it is the one quantity sharding can perturb.
PER=$(( $(nproc) / SHARDS ))
for i in $(seq 0 $((SHARDS - 1))); do
    LO=$(( i * PER ))
    HI=$(( LO + PER - 1 ))
    OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$LO-$HI" \
        "$BIN" "${ARGS[@]}" "$OUT/shard$i.csv" "${TAIL[@]}" \
        >"$OUT/shard$i.log" 2>"$OUT/shard$i.err" &
done
wait

head -1 "$OUT/shard0.csv" >"$OUT/voxel3mm_nomargin.csv"
for i in $(seq 0 $((SHARDS - 1))); do
    tail -n +2 "$OUT/shard$i.csv" >>"$OUT/voxel3mm_nomargin.csv"
done
echo "rows: $(tail -n +2 "$OUT/voxel3mm_nomargin.csv" | wc -l)"
