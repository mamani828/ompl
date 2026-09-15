#!/usr/bin/env bash
# D2' -- the cell the matrix was missing: EXACT field with an explicit 3 mm reserve.
#
# `interpolationBuffer()` derives 3 mm from a 3 mm voxel and 0 from the exact field, so
# the only way to have both exactness and a reserve is to pass the buffer by hand. Never
# run before tonight: grid+0, grid+3mm, exact+0 and exact+1mm all exist, exact+3mm did not.
#
# Two reasons to prefer it over the grid version:
#   - On the exact field the 3 mm is PURE conservatism. A grid buffer partly goes on
#     covering the interpolant's own disagreement with the function it sampled.
#   - The grid audits leniently. Distance to a convex set is convex, trilinear
#     interpolation of a convex function lies ABOVE it, and UR5MBMBenchmark hands the same
#     field to `audited` and `guard` -- so the grid's near-zero unsafe count is partly the
#     auditor sharing the planner's error. Exact audits against the geometry.
#
# Pair margin at k=0.25 (diagnostic, see the sweep run), so this measures whether 3 mm of
# honest reserve removes the world penetration that D1 showed at zero reserve
# (52-53 rows, -16.5 mm, every one environment and not one self-collision).
set -uo pipefail
cd "$(dirname "$0")/../.."
SHARDS=24
BIN=./build/demos/demo_UR5MBMBenchmark
OUT=results/mbm_pairfix_all689_20260914
export OMPL_MBM_EXACT_FIELD=1 OMPL_CBF_SCREENING=1 OMPL_CBF_MAX_SPEED=10 OMPL_CBF_JOINT_LIMITS=0
export OMPL_UR5_SELF_PAIR_MARGIN_SCALE=0.25 OMPL_MBM_SELF_BUFFER=0.001
export OMPL_MBM_ROWS=isSafe,qpPlain,qpFixed,qpAdaptive,qpFreeGate,qpEnvelope,qpFreeEnvelope
for i in $(seq 0 $((SHARDS-1))); do
  OMPL_MBM_SHARD=$i/$SHARDS taskset -c "$i" \
    "$BIN" "$OUT/scenes_all.txt" 100 10.0 0.003 0.001 2.0 0.0 0.003 -1 20 -1 0.0 "" -1 1 \
    "$OUT/d2exact_shard$i.csv" 1 0 4 0 0 10000 \
    >"$OUT/d2exact_shard$i.log" 2>"$OUT/d2exact_shard$i.err" &
done
wait
head -1 "$OUT/d2exact_shard0.csv" >"$OUT/d2exact_3mm.csv"
for i in $(seq 0 $((SHARDS-1))); do tail -n +2 "$OUT/d2exact_shard$i.csv" >>"$OUT/d2exact_3mm.csv"; done
echo "d2exact_3mm rows: $(tail -n +2 "$OUT/d2exact_3mm.csv" | wc -l)"
