#!/bin/bash
set -euo pipefail

# Step 1: Install pip-only dependencies
pip install pytorch-lightning==1.8.6 transformers==4.26.1 hydra-core omegaconf

# Step 2: Install flash_attn CPU stub as flash_attn
python "${SRC_DIR}/install_flash_attn_stub.py"

# Step 3: Install the main package
"${PYTHON}" -m pip install . --no-deps -vv
