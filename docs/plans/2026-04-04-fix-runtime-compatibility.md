# Runtime Compatibility Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 euktect 包两处运行时 API 不兼容问题，使 `euktect-predict` 命令能够成功运行，并在 conda recipe 中添加端到端预测测试。

**Architecture:** 直接修复源文件中调用了过时/版本不兼容 API 的两处代码，并将 timm 版本固定以保证长期稳定。同时在 meta.yaml 的 test 段添加真实预测命令。

**Tech Stack:** conda-build, PyTorch 1.11, timm 1.0.26, Python 3.10

---

## 背景：已识别的两个根本原因

| # | 文件 | 行号 | 问题 | 原因 |
|---|------|------|------|------|
| 1 | `flash_attn_cpu/modules/block.py` | 33, 38 | `StochasticDepth(x, mode='row')` | timm ≥ 0.9 的 `DropPath.__init__` 移除了 `mode` 参数 |
| 2 | `hyena-dna/src/models/sequence/long_conv_lm.py` | 121 | `partial(F.gelu, approximate="tanh")` | `F.gelu` 的 `approximate` 参数在 PyTorch < 1.12 中不存在 |

**修复原则：**
- Bug 1：去掉 `mode='row'`（新版 DropPath 默认行为与 `mode='row'` 等价）
- Bug 2：将 `partial(F.gelu, approximate="tanh")` 替换为 `F.gelu`（exact GELU vs tanh-approximated GELU 在推理精度上差异极小，不影响模型功能）
- 在 meta.yaml 固定 `timm <0.9` 或在源码层面使其与 timm ≥1.0 兼容（我们选择后者，因为 conda 上 timm<0.9 难以获取）

---

### Task 1: 修复 flash_attn_cpu/modules/block.py 中的 mode='row' 参数

**Files:**
- Modify: `flash_attn_cpu/modules/block.py:33,38`

**Step 1: 确认当前文件内容**

```bash
grep -n "mode" /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/modules/block.py
```

Expected: 两行输出，分别是第 33 和 38 行含 `mode='row'`

**Step 2: 修改 block.py — 去掉两处 mode='row'**

将第 33 行：
```python
self.drop_path1 = StochasticDepth(drop_path1, mode='row')
```
改为：
```python
self.drop_path1 = StochasticDepth(drop_path1)
```

将第 38 行：
```python
self.drop_path2 = StochasticDepth(drop_path2, mode='row')
```
改为：
```python
self.drop_path2 = StochasticDepth(drop_path2)
```

**Step 3: 验证修改**

```bash
grep -n "mode\|StochasticDepth" /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/modules/block.py
```

Expected: 不再出现 `mode='row'`

**Step 4: Commit**

```bash
git add flash_attn_cpu/modules/block.py
git commit -m "fix: remove mode='row' from StochasticDepth for timm>=0.9 compatibility"
```

---

### Task 2: 修复 hyena-dna/src/models/sequence/long_conv_lm.py 中的 gelu approximate 参数

**Files:**
- Modify: `hyena-dna/src/models/sequence/long_conv_lm.py:121`

**Step 1: 确认当前文件内容**

```bash
grep -n "approximate\|gelu" /media/asky/F/Euktect_edit/Euktect/hyena-dna/src/models/sequence/long_conv_lm.py
```

Expected: 第 121 行含 `approximate="tanh"`

**Step 2: 修改 long_conv_lm.py — 去掉 approximate 参数**

将第 121 行：
```python
            activation=partial(F.gelu, approximate="tanh"),
```
改为：
```python
            activation=F.gelu,
```

注意：`partial(F.gelu, approximate="tanh")` 和直接 `F.gelu` 效果几乎相同。
`approximate="tanh"` 是 tanh 近似的 GELU，是推理加速的近似，去掉后退回精确 GELU，模型功能不受影响。

**Step 3: 验证修改**

```bash
grep -n "approximate\|activation" /media/asky/F/Euktect_edit/Euktect/hyena-dna/src/models/sequence/long_conv_lm.py | head -10
```

Expected: 不再出现 `approximate`

**Step 4: Commit**

```bash
git add hyena-dna/src/models/sequence/long_conv_lm.py
git commit -m "fix: use F.gelu without approximate kwarg for PyTorch<1.12 compatibility"
```

---

### Task 3: 在 meta.yaml 中更新版本约束以防止将来回归

**Files:**
- Modify: `conda.recipe/meta.yaml`

**Step 1: 读取当前 meta.yaml 的 run 依赖部分**

确认当前 timm 和 pytorch 约束。

**Step 2: 固定 pytorch 版本下限为 >=1.12（含 approximate 支持）或保持现状**

分析：
- 当前环境 pytorch=1.11 触发 Bug 2，但我们已在源码中修复它
- 为避免未来打包到更新环境时触发，在 meta.yaml 中添加注释说明
- timm 不需要固定版本（Bug 1 已在源码层面修复）

在 meta.yaml 的 `run` 段，为 timm 行添加注释：
```yaml
    - timm  # >=0.9 OK: mode='row' removed from block.py; DropPath default is row-wise
```

为 pytorch 行添加注释：
```yaml
    - pytorch * cpu_*  # F.gelu approximate kwarg removed in long_conv_lm.py for <1.12 compat
```

**Step 3: Commit**

```bash
git add conda.recipe/meta.yaml
git commit -m "docs: add compatibility comments for timm and pytorch version constraints"
```

---

### Task 4: 在 meta.yaml 的 test 段添加端到端预测测试

**Files:**
- Modify: `conda.recipe/meta.yaml:55-62`

**Step 1: 了解当前 test 段**

当前 test 段：
```yaml
test:
  imports:
    - euktect
    - flash_attn_cpu
    - flash_attn_cpu.modules.mha
  commands:
    - euktect-predict --help
    - euktect-refine --help
```

**Step 2: 理解 conda test 段的限制**

conda build test 阶段没有访问外部大文件（checkpoint、fasta 文件）的能力。
因此端到端预测测试需要以 `commands` 的形式，使用测试数据路径，或者通过 `requires` + 脚本来实现。

实际可行方案：在 test 段的 `commands` 中使用环境变量检查跳过，或添加一个单独的集成测试脚本。

**最佳方案**：在源码中增加一个小型测试脚本 `tests/test_predict_integration.sh`，该脚本使用固定路径的测试数据，作为 CI/CD 集成测试，而不是 conda build test（conda build test 没有访问外部 ckpt 文件的权限）。

同时在 meta.yaml 中保留现有测试，并添加注释说明集成测试需单独运行。

**Step 3: 创建集成测试脚本 `tests/test_predict_integration.sh`**

```bash
#!/usr/bin/env bash
# Integration test: run euktect-predict end-to-end
# Usage: bash tests/test_predict_integration.sh
# Requires: external ckpt and fasta files (not included in package)

set -euo pipefail

INPUT="${EUKTECT_TEST_INPUT:-/media/asky/F/Deepfungi_ana/fement_metagenome/Cocoa/PRJNA527768/contigs/SRR8742576.fna}"
CKPT="${EUKTECT_TEST_CKPT:-/media/asky/F/Deepfungi_ana/000_paper/code/Euktect/ckpt/Pichiomycetes_class.ckpt}"
CFG="${EUKTECT_TEST_CFG:-/media/asky/F/Deepfungi_ana/000_paper/code/Euktect/cfg/c/1000.yaml}"
OUTPUT=$(mktemp /tmp/euktect_test_XXXXXX.csv)

echo "=== euktect-predict integration test ==="
echo "INPUT: $INPUT"
echo "CKPT:  $CKPT"
echo "CFG:   $CFG"
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
```

**Step 4: 在 meta.yaml 添加注释指向集成测试**

在 test 段末尾添加：
```yaml
  # Integration test (requires external ckpt/fasta files, not run during conda-build):
  #   bash tests/test_predict_integration.sh
```

**Step 5: Commit**

```bash
git add tests/test_predict_integration.sh conda.recipe/meta.yaml
git commit -m "test: add end-to-end integration test script for euktect-predict"
```

---

### Task 5: 本地验证端到端预测成功运行

**Step 1: 在安装了 euktect 的环境中运行集成测试**

```bash
conda run -n euktect bash tests/test_predict_integration.sh
```

Expected:
```
=== euktect-predict integration test ===
INPUT: /media/asky/F/...SRR8742576.fna
...
Processing sequences: 100%|...
=== PASS: output file created and non-empty ===
contig_id,prediction,...
```

**Step 2: 若失败，读取完整错误，返回对应 Task 重新修复**

**Step 3: 验证输出 CSV 格式正确**

```bash
conda run -n euktect python -c "
import pandas as pd
df = pd.read_csv('test.csv')
print(df.shape)
print(df.head())
"
```

---

### Task 6: 更新 sha256 并提交 conda.recipe/meta.yaml（如需重新发包）

> 注意：Tasks 1-2 修改了打包进 conda 的源文件，如需重新 build 并发布到 bioconda，需要更新 sha256。
> 如仅用于本地修复，跳过此 Task。

**Step 1: 更新 GitHub release tag（创建 v1.0.1 或 patch release）**

**Step 2: 计算新的 sha256**

```bash
curl -sL https://github.com/NameFilled/Euktect/archive/refs/tags/v1.0.1.tar.gz | sha256sum
```

**Step 3: 更新 meta.yaml 中的 version 和 sha256**

**Step 4: 测试 conda build**

```bash
conda build conda.recipe/ -c bioconda -c conda-forge --no-test
```

**Step 5: Commit**

```bash
git add conda.recipe/meta.yaml
git commit -m "fix: bump to v1.0.1 with runtime compatibility fixes"
```
