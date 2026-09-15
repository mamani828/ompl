#!/usr/bin/env bash
# Fair all-689 comparison with the vanilla no-QP accept-or-stop controller.
#
# Fairness controls:
# - Every OMPL row receives the same locally seeded per-problem sample stream. The
#   benchmark derives it from the problem's absolute ordinal, so 24-way sharding does
#   not change the stream. RRTConnect's otherwise constructed RNG is unused.
# - Every row uses the same 0.1 rad goal tolerance, 2.0 rad range, 10 s time limit,
#   problem eligibility test, exact analytic field, and audited safety model.
# - The baseline validity-check resolution is inferred from the 0.001 s rollout step.
# - OMPL_QPFREE_REPAIR=0 makes both no-QP rows accept the nominal control unchanged or
#   stop; neither row scales/brakes or projects the control.
set -uo pipefail
cd "$(dirname "$0")/../.."

SHARDS=24
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_all689_k05_acceptstop_fair_20260915
SCENES=results/mbm_all689_k05_20260915/scenes_all.txt

export OMPL_MBM_EXACT_FIELD=1
export OMPL_CBF_VERIFY_STEP=2
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.5
export OMPL_MBM_SELF_BUFFER=0
export OMPL_CBF_SCREENING=1
export OMPL_CBF_MAX_SPEED=10
export OMPL_CBF_JOINT_LIMITS=0
export OMPL_QPFREE_REPAIR=0
export OMPL_MBM_GOAL_TOLERANCE=0.1
export OMPL_SDF_BAKE_THREADS=1
export OMPL_MBM_ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope,VAMP

mkdir -p "$OUT"
git rev-parse HEAD >"$OUT/git_commit.txt"
git status --short >"$OUT/git_status.txt"
git diff -- demos/UR5MBMBenchmark.cpp demos/UR5QPFreeGate.h demos/RobotQPFreeGate.h \
  src/ompl/cbf/ControlFilter.h src/ompl/cbf/src/CBFControlFilter.cpp >"$OUT/source.diff"
sha256sum "$BIN" >"$OUT/binary.sha256"
env | sort | grep -E '^OMPL_(MBM|CBF|UR5|QPFREE|SDF)_' >"$OUT/environment.txt"

pids=()
for i in $(seq 0 $((SHARDS-1))); do
  OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
    "$BIN" "$SCENES" 100 10.0 0.003 0.001 2.0 0.0 -1 -1 20 -1 0.0 "" -1 1 \
    "$OUT/shard$i.csv" 1 0 4 0 0 10000 >"$OUT/shard$i.log" 2>"$OUT/shard$i.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "one or more shards failed" >&2
  exit 1
fi

head -1 "$OUT/shard0.csv" >"$OUT/all689_k05_acceptstop_fair.csv"
for i in $(seq 0 $((SHARDS-1))); do
  tail -n +2 "$OUT/shard$i.csv" >>"$OUT/all689_k05_acceptstop_fair.csv"
done
echo "rows: $(tail -n +2 "$OUT/all689_k05_acceptstop_fair.csv" | wc -l)"
