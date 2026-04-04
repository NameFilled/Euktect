#!/usr/bin/env bash
# Integration test: run euktect-predict end-to-end
# Usage: bash tests/test_predict_integration.sh
# Requires: external ckpt and fasta files (not included in package)
# Override defaults with environment variables:
#   EUKTECT_TEST_INPUT, EUKTECT_TEST_CKPT, EUKTECT_TEST_CFG

set -euo pipefail

INPUT="${EUKTECT_TEST_INPUT:-/media/asky/F/Deepfungi_ana/fement_metagenome/Cocoa/PRJNA527768/contigs/SRR8742576.fna}"
CKPT="${EUKTECT_TEST_CKPT:-/media/asky/F/Deepfungi_ana/000_paper/code/Euktect/ckpt/Pichiomycetes_class.ckpt}"
CFG="${EUKTECT_TEST_CFG:-/media/asky/F/Deepfungi_ana/000_paper/code/Euktect/cfg/c/1000.yaml}"
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
