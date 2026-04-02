@echo off
setlocal enableextensions

:: Step 1: Install pip-only dependencies
pip install pytorch-lightning==1.8.6 transformers==4.26.1 hydra-core omegaconf
if errorlevel 1 exit 1

:: Step 2: Install flash_attn CPU stub
python "%SRC_DIR%\install_flash_attn_stub.py"
if errorlevel 1 exit 1

:: Step 3: Install main package
"%PYTHON%" -m pip install . --no-deps -vv
if errorlevel 1 exit 1
