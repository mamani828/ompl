#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

BASE=results/mbm_voxel3mm_step0001_qprows_all689_nomargins_20260914/voxel3mm_step0001_qprows_all689_nomargins.csv
EXTRA=results/mbm_voxel3mm_step0001_qpfree_envelope_all689_20260914/voxel3mm_step0001_qpfree_envelope_all689.csv
STRICT=results/mbm_voxel3mm_step0001_tron_noqp_all689_20260914/voxel3mm_step0001_tron_noqp_all689.csv
OUT=results/mbm_voxel3mm_step0001_qprows_all689_nomargins_20260914/voxel3mm_step0001_qprows_all689_nomargins_seven_methods.csv

grep -v ",qpFreeGate," "$BASE" > "$OUT"
tail -n +2 "$STRICT" >> "$OUT"
tail -n +2 "$EXTRA" >> "$OUT"
