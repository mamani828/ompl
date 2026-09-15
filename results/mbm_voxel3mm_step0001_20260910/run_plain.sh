#!/usr/bin/env bash
# UR5 MotionBenchMaker at a 3 mm voxel, no margin, 1 ms controller step.
# Companion to mbm_voxel3mm_nomargin_20260910 (10 ms step): the fine step drops the
# one-step floor to 0.01 rad so the hold certificate actually binds (at-region hops
# 6% -> 24%), which is what gives envelope-vs-L1 any resolution.
#
# Plain CBF-RRT row only (qpPlain), run separately so the completed five-row shards
# are preserved.
# Screening on, RRT-Connect range 2.0 rad, margin 0 and filter buffer 0.
#
# Sharded over the problem set: each problem keeps the seed it would have had in a
# single-process run, so `cat` of the shard CSVs is the unsharded CSV. One 3 mm field
# is ~3.7 GB, so SHARDS is bounded by memory, not cores.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=${SHARDS:-7}
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_voxel3mm_step0001_20260910
SCENES=$OUT/scenes_all.txt

export OMPL_CBF_SCREENING=1
export OMPL_MBM_ROWS=qpPlain
export OMPL_SDF_BAKE_THREADS=$(( $(nproc) / SHARDS ))
# Lift the Constant(0.5) placeholder cap so it is not the binding constraint. It cannot
# be zero: FilteredStateSpace rejects non-positive speeds and the QP needs a bounded
# control box. See effectiveMaxSpeed().
export OMPL_CBF_MAX_SPEED=10
# Off: the geometric space already bounds sampled configurations, so this isolates the
# effect of additionally constraining one Euler step inside the QP.
export OMPL_CBF_JOINT_LIMITS=0

# segFrac -1 ties RRT-Connect's edge-checking spacing to the rollout step, so both rows
# are scored at the same coarseness: maxSpeed 10 x stepSize 0.001 = 0.0100 rad. That is
# only sane because the step is fine here -- at stepSize 0.01 the same rule would give
# 0.1 rad, coarse enough for the baseline to tunnel through a 20 mm shelf panel.
#          scenes perScene secs voxel  step range margin buffer segFrac kappa maxStep selfMargin prefix shortcut seed
ARGS=("$SCENES"      100  10.0 0.003 0.001   2.0    0.0  0.003      -1    20      -1        0.0     ""       -1    1 )
#  csvPath is appended per shard below, then: safeHops picardIter picardWindow picardWorkers trajPrefixes rolloutBudget
# rolloutCallBudget 10000: at 0.01 rad per hop that is 100 rad per extension, far past
# the 20 rad the range allows, so the budget never binds. At 1000 it would have capped
# an extension at 10 rad -- half the range.
TAIL=( 1 0 4 0 0 10000 )

# Pin each shard to its own cores. Planning is single-threaded, so a pinned shard gets
# a dedicated core for it and the shards cannot steal each other's throughput. That
# matters for rows that hit the time limit: their sample count is bounded by wall-clock
# progress rather than by the seed, so it is the one quantity sharding can perturb.
PER=$(( $(nproc) / SHARDS ))
for i in $(seq 0 $((SHARDS - 1))); do
    LO=$(( i * PER ))
    HI=$(( LO + PER - 1 ))
    OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$LO-$HI" \
        "$BIN" "${ARGS[@]}" "$OUT/plain_shard$i.csv" "${TAIL[@]}" \
        >"$OUT/plain_shard$i.log" 2>"$OUT/plain_shard$i.err" &
done
wait

head -1 "$OUT/plain_shard0.csv" >"$OUT/qp_plain.csv"
for i in $(seq 0 $((SHARDS - 1))); do
    tail -n +2 "$OUT/plain_shard$i.csv" >>"$OUT/qp_plain.csv"
done
echo "rows: $(tail -n +2 "$OUT/qp_plain.csv" | wc -l)"
