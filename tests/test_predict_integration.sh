#!/usr/bin/env bash
# Integration test: run euktect-predict end-to-end
# Usage: bash tests/test_predict_integration.sh
# Requires: ckpt and cfg files at the paths below

set -euo pipefail

INPUT="${EUKTECT_TEST_INPUT:-/media/asky/F/Euktect_edit/Euktect/tests/test.fna}"
CKPT="${EUKTECT_TEST_CKPT:-/media/asky/F/Euktect_edit/Euktect/tests/1000.ckpt}"
CFG="${EUKTECT_TEST_CFG:-/media/asky/F/Euktect_edit/Euktect/tests/1000.yaml}"
OUTPUT=$(mktemp /tmp/euktect_test_XXXXXX.csv)

echo "=== euktect-predict integration test ==="
echo "INPUT:  $INPUT"
echo "CKPT:   $CKPT"
echo "CFG:    $CFG"
echo "OUTPUT: $OUTPUT"

euktect-predict --input "$INPUT" --ckpt "$CKPT" --cfg "$CFG" --output "$OUTPUT"

if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    echo "=== PASS: output file created and non-empty ==="
    head -3 "$OUTPUT"
    rm -f "$OUTPUT"
else
    echo "=== FAIL: output file missing or empty ==="
    rm -f "$OUTPUT"
    exit 1
fi
