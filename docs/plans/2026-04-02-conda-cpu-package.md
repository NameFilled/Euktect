# Euktect CPU-Only Conda Package Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Euktect + HyenaDNA 项目打包为一个功能完整、跨平台兼容的 CPU-only conda 包，使用户无需 GPU 即可运行 DNA 序列预测（predict.py）和 MAG 精炼（refine.py）。

**Architecture:** 
- 创建一个 `flash_attn` CPU 兼容 stub 包，替代 CUDA 专用的 flash-attention 库
- 修复 `block_fft.py` 中硬编码的 `.cuda()` 调用，改为设备无关写法
- 用 conda-build meta.yaml + setup.py 打包整个项目，指定 CPU-only 的 PyTorch

**Tech Stack:** Python 3.8+, PyTorch (CPU-only), conda-build, pyfaidx, einops, transformers==4.26.1, pytorch-lightning==1.8.6

---

## 分析摘要

### GPU 依赖清单

| 文件 | GPU 依赖 | 处理方案 |
|------|---------|---------|
| `hyena-dna/src/models/sequence/long_conv_lm.py` | `flash_attn.modules.{mha,mlp,block,embedding}`, `flash_attn.utils.*` | flash_attn CPU stub |
| `hyena-dna/src/models/sequence/dna_embedding.py` | `flash_attn.utils.generation`, `flash_attn.utils.distributed` | flash_attn CPU stub |
| `hyena-dna/src/models/sequence/hyena.py` | `flash_attn.ops.fused_dense.FusedDense`（可选 import） | 已有 try/except，stub 即可 |
| `hyena-dna/src/models/sequence/block_fft.py` | `.cuda()` 硬编码 | 改为 `torch.device` 方式 |
| `hyena-dna/src/tasks/torchmetrics.py` | `flash_attn.losses.cross_entropy`（可选） | flash_attn CPU stub |
| `predict.py` | `device = "cuda" if torch.cuda.is_available() else "cpu"` | 已自动降级，无需修改 |

### simple_lm.py 中已有可参考的 CPU 实现
- `Mlp`（第 191 行）：标准 nn.Linear 实现
- `Block`（第 213 行）：标准 Transformer Block 实现
- 这些将作为 flash_attn CPU stub 的实现参考

---

## Task 1: 创建 flash_attn CPU stub 包骨架

**Files:**
- Create: `flash_attn_cpu/__init__.py`
- Create: `flash_attn_cpu/modules/__init__.py`
- Create: `flash_attn_cpu/utils/__init__.py`
- Create: `flash_attn_cpu/ops/__init__.py`
- Create: `flash_attn_cpu/losses/__init__.py`

**Step 1: 创建目录结构**

```bash
mkdir -p /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/modules
mkdir -p /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/utils
mkdir -p /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/ops
mkdir -p /media/asky/F/Euktect_edit/Euktect/flash_attn_cpu/losses
```

**Step 2: 创建各 `__init__.py`（内容均为空文件）**

对以下文件分别 Write 空内容：
- `flash_attn_cpu/__init__.py`
- `flash_attn_cpu/modules/__init__.py`
- `flash_attn_cpu/utils/__init__.py`
- `flash_attn_cpu/ops/__init__.py`
- `flash_attn_cpu/losses/__init__.py`

**Step 3: Commit**

```bash
git add flash_attn_cpu/
git commit -m "feat: add flash_attn CPU stub package skeleton"
```

---

## Task 2: 实现 flash_attn.modules.mha（MHA）

**Files:**
- Create: `flash_attn_cpu/modules/mha.py`

用标准 PyTorch `nn.MultiheadAttention` 实现 `MHA`，兼容 flash_attn MHA 的接口签名。

**Step 1: 查看原 flash_attn MHA 在代码中的使用方式**

在 `long_conv_lm.py` 中，MHA 用法如下：
```python
mixer_cls = partial(MHA, causal=True, layer_idx=0, ...)
# 然后：self.mixer = mixer_cls(dim)
```
即：`MHA(dim, causal=True, layer_idx=0, num_heads=..., ...)`

**Step 2: 实现 mha.py**

```python
# flash_attn_cpu/modules/mha.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MHA(nn.Module):
    """CPU-compatible Multi-Head Attention stub for flash_attn.modules.mha.MHA."""

    def __init__(self, embed_dim, num_heads=None, causal=False, layer_idx=None,
                 bias=True, dropout=0.0, softmax_scale=None,
                 fused_bias_fc=False, dwconv=False, rotary_emb_dim=0,
                 rotary_emb_scale_base=0, rotary_emb_interleaved=False,
                 use_flash_attn=False, checkpointing=False, return_residual=False,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        if num_heads is None:
            num_heads = max(1, embed_dim // 64)
        self.num_heads = num_heads
        self.causal = causal
        self.return_residual = return_residual
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None,
                max_seqlen=None, mixer_subset=None, inference_params=None, **kwargs):
        # x: (batch, seqlen, embed_dim)
        B, L, D = x.shape
        head_dim = D // self.num_heads

        qkv = self.Wqkv(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for attention: (B, heads, L, head_dim)
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)

        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1
            )

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(attn_out)
        return out if not self.return_residual else (out, x)


# ParallelMHA: in CPU-only context, parallel is not needed
ParallelMHA = MHA
```

**Step 3: Commit**

```bash
git add flash_attn_cpu/modules/mha.py
git commit -m "feat: implement CPU-compatible MHA stub"
```

---

## Task 3: 实现 flash_attn.modules.mlp 和 flash_attn.modules.block

**Files:**
- Create: `flash_attn_cpu/modules/mlp.py`
- Create: `flash_attn_cpu/modules/block.py`

直接参考 `hyena-dna/src/models/sequence/simple_lm.py` 中第 191-302 行已有的 CPU 实现。

**Step 1: 实现 mlp.py**

```python
# flash_attn_cpu/modules/mlp.py
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    """CPU-compatible Mlp, mirroring flash_attn.modules.mlp.Mlp interface."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=F.gelu, return_residual=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

# FusedMLP and ParallelFusedMLP: degenerate to regular Mlp on CPU
FusedMLP = Mlp
ParallelFusedMLP = Mlp
```

**Step 2: 实现 block.py**

参考 simple_lm.py Block 实现（第 213-302 行），保持与 flash_attn Block 相同的接口：

```python
# flash_attn_cpu/modules/block.py
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath as StochasticDepth

from flash_attn_cpu.modules.mha import MHA
from flash_attn_cpu.modules.mlp import Mlp

class Block(nn.Module):
    """CPU-compatible Block, mirroring flash_attn.modules.block.Block."""

    def __init__(self, dim, mixer_cls=None, mlp_cls=None, norm_cls=nn.LayerNorm,
                 dropout_cls=nn.Dropout, prenorm=True, resid_dropout1=0., resid_dropout2=0.,
                 drop_path1=0., drop_path2=0., return_residual=False,
                 residual_in_fp32=False, fused_dropout_add_ln=False):
        super().__init__()
        self.prenorm = prenorm
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=max(1, dim // 64))
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode='row')
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode='row')
            self.norm2 = norm_cls(dim)

    def forward(self, hidden_states, residual=None, mixer_subset=None, mixer_kwargs=None):
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                dropped = self.drop_path2(self.dropout2(hidden_states))
                residual = (dropped + residual)
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_kwargs = mixer_kwargs or {}
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)
            hidden_states = self.norm1(hidden_states)
            if not isinstance(self.mlp, nn.Identity):
                hidden_states = self.mlp(hidden_states)
                hidden_states = self.norm2(hidden_states)
            return hidden_states
```

**Step 3: Commit**

```bash
git add flash_attn_cpu/modules/mlp.py flash_attn_cpu/modules/block.py
git commit -m "feat: implement CPU-compatible Mlp and Block stubs"
```

---

## Task 4: 实现 flash_attn.modules.embedding 和 utils

**Files:**
- Create: `flash_attn_cpu/modules/embedding.py`
- Create: `flash_attn_cpu/utils/generation.py`
- Create: `flash_attn_cpu/utils/distributed.py`
- Create: `flash_attn_cpu/ops/fused_dense.py`
- Create: `flash_attn_cpu/ops/layer_norm.py`
- Create: `flash_attn_cpu/losses/cross_entropy.py`

**Step 1: 实现 embedding.py**

```python
# flash_attn_cpu/modules/embedding.py
import torch
import torch.nn as nn

class GPT2Embeddings(nn.Module):
    """CPU-compatible GPT2Embeddings."""
    def __init__(self, embed_dim, vocab_size, max_position_embeddings=0,
                 padding_idx=None, word_embed_proj_dim=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if word_embed_proj_dim is None:
            word_embed_proj_dim = embed_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
                                             padding_idx=padding_idx, **factory_kwargs)
        self.project_in = (nn.Linear(word_embed_proj_dim, embed_dim, bias=False, **factory_kwargs)
                           if word_embed_proj_dim != embed_dim else None)
        self.max_position_embeddings = max_position_embeddings
        if max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                     **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            embeddings = embeddings + self.position_embeddings(position_ids)
        return embeddings

# ParallelGPT2Embeddings: same as GPT2Embeddings on CPU
ParallelGPT2Embeddings = GPT2Embeddings
```

**Step 2: 实现 utils/generation.py**

```python
# flash_attn_cpu/utils/generation.py
class GenerationMixin:
    """No-op GenerationMixin for CPU inference."""
    pass
```

**Step 3: 实现 utils/distributed.py**

```python
# flash_attn_cpu/utils/distributed.py
def sync_shared_params(module, process_group):
    """No-op: no distributed sync on CPU-only single-process inference."""
    pass

def all_gather_raw(input_, process_group, async_op=False):
    """No-op: returns input unchanged."""
    return input_, None
```

**Step 4: 实现 ops/fused_dense.py**

```python
# flash_attn_cpu/ops/fused_dense.py
import torch.nn as nn

class ColumnParallelLinear(nn.Linear):
    """CPU fallback: standard nn.Linear."""
    def __init__(self, in_features, out_features, process_group=None,
                 bias=True, sequence_parallel=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

class FusedDense(nn.Linear):
    """CPU fallback: standard nn.Linear."""
    pass
```

**Step 5: 实现 ops/layer_norm.py**

```python
# flash_attn_cpu/ops/layer_norm.py
import torch
import torch.nn.functional as F

def dropout_add_layer_norm(x0, residual, weight, bias, dropout_p, epsilon,
                            rowscale=None, layerscale=None, prenorm=False,
                            x0_dtype=None, return_dropout_mask=False):
    """CPU fallback: unfused dropout + add + layernorm."""
    x = F.dropout(x0.float(), p=dropout_p, training=False)
    if residual is not None:
        x = x + residual.float()
    out = F.layer_norm(x, weight.shape, weight.float(), bias.float(), epsilon)
    out = out.to(x0.dtype)
    if prenorm:
        return out, x
    return out
```

**Step 6: 实现 losses/cross_entropy.py**

```python
# flash_attn_cpu/losses/cross_entropy.py
import torch.nn as nn

class CrossEntropyLoss(nn.CrossEntropyLoss):
    """CPU fallback: standard PyTorch CrossEntropyLoss."""
    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0,
                 inplace_backward=False, **kwargs):
        super().__init__(ignore_index=ignore_index, reduction=reduction,
                         label_smoothing=label_smoothing)
```

**Step 7: Commit**

```bash
git add flash_attn_cpu/
git commit -m "feat: implement remaining flash_attn CPU stubs (embedding, utils, ops, losses)"
```

---

## Task 5: 修复 block_fft.py 中的硬编码 .cuda() 调用

**Files:**
- Modify: `hyena-dna/src/models/sequence/block_fft.py:15,24,25`

**Step 1: 读取文件确认行号**

Read `hyena-dna/src/models/sequence/block_fft.py` 查看第 10-30 行。

**Step 2: 修改硬编码 .cuda() 调用**

将：
```python
n = torch.arange(N).cuda()
```
改为：
```python
n = torch.arange(N)
```
（移除 `.cuda()`，PyTorch 的 FFT 操作在 CPU 上原生支持）

同样修改第 24-25 行：
```python
n_a = torch.arange(n).cuda().view(-1, 1)
m_a = torch.arange(m).cuda()
```
改为：
```python
n_a = torch.arange(n).view(-1, 1)
m_a = torch.arange(m)
```

**Step 3: 验证修改**

```bash
grep -n "\.cuda()" /media/asky/F/Euktect_edit/Euktect/hyena-dna/src/models/sequence/block_fft.py
# 期望：没有输出
```

**Step 4: Commit**

```bash
git add hyena-dna/src/models/sequence/block_fft.py
git commit -m "fix: remove hardcoded .cuda() in block_fft.py for CPU compatibility"
```

---

## Task 6: 创建 setup.py 和包结构

**Files:**
- Create: `setup.py`
- Create: `euktect/__init__.py`（包入口）

**Step 1: 创建 setup.py**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="euktect",
    version="1.0.0",
    description="Eukaryotic genome assessment tool using HyenaDNA (CPU-only conda package)",
    packages=find_packages(exclude=["docs", "tests"]),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "rich",
        "pytorch-lightning==1.8.6",
        "hydra-core",
        "omegaconf",
        "einops",
        "opt_einsum",
        "transformers==4.26.1",
        "timm",
        "pyfaidx",
        "polars",
        "loguru",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "euktect-predict=predict:main",
            "euktect-refine=refine:main",
        ],
    },
)
```

**Step 2: 创建 euktect/__init__.py**

```python
# euktect/__init__.py
"""Euktect: Eukaryotic MAG assessment using HyenaDNA."""
__version__ = "1.0.0"
```

**Step 3: Commit**

```bash
git add setup.py euktect/__init__.py
git commit -m "feat: add setup.py and package entry point"
```

---

## Task 7: 创建 conda meta.yaml 构建配方

**Files:**
- Create: `conda.recipe/meta.yaml`

**Step 1: 创建 meta.yaml**

```yaml
{% set version = "1.0.0" %}

package:
  name: euktect
  version: {{ version }}

source:
  path: ..

build:
  number: 0
  script: {{ PYTHON }} -m pip install . --no-deps -vv
  noarch: python

requirements:
  host:
    - python >=3.8
    - pip
    - setuptools

  run:
    - python >=3.8
    - pytorch >=1.13,<2.1 [cpu]  # CPU-only PyTorch
    - cpuonly                     # conda-forge meta-package to ensure no CUDA torch
    - numpy
    - scipy
    - pandas
    - scikit-learn
    - matplotlib
    - tqdm
    - rich
    - einops
    - opt_einsum
    - timm
    - pyfaidx
    - polars
    - loguru
    - pyyaml
    - pip:
      - pytorch-lightning==1.8.6
      - hydra-core
      - omegaconf
      - transformers==4.26.1

test:
  imports:
    - euktect

about:
  home: https://github.com/NameFilled/Euktect
  license: MIT
  summary: Eukaryotic MAG assessment using HyenaDNA (CPU-only)
```

**注意：** CPU-only PyTorch 在 conda 中通过 `cpuonly` 包强制，需使用 `pytorch` channel：
```
-c pytorch -c conda-forge
```

**Step 2: Commit**

```bash
git add conda.recipe/meta.yaml
git commit -m "feat: add conda build recipe (CPU-only)"
```

---

## Task 8: 创建构建脚本和 flash_attn 安装脚本

**Files:**
- Create: `conda.recipe/build.sh`
- Create: `conda.recipe/bld.bat`
- Create: `install_flash_attn_stub.py`（用于将 flash_attn_cpu 作为 flash_attn 安装）

**Step 1: 实现 install_flash_attn_stub.py**

这个脚本将 flash_attn_cpu 目录以 `flash_attn` 名称安装到 Python 环境中：

```python
#!/usr/bin/env python3
"""Install flash_attn_cpu stub as flash_attn in current Python environment."""
import os
import sys
import shutil
import site

def main():
    src = os.path.join(os.path.dirname(__file__), "flash_attn_cpu")
    site_packages = site.getsitepackages()[0]
    dest = os.path.join(site_packages, "flash_attn")
    if os.path.exists(dest):
        print(f"flash_attn already exists at {dest}, removing...")
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"Installed flash_attn CPU stub to {dest}")

if __name__ == "__main__":
    main()
```

**Step 2: 创建 build.sh**

```bash
#!/bin/bash
set -e

# Install flash_attn CPU stub as flash_attn
python install_flash_attn_stub.py

# Install the main package
$PYTHON -m pip install . --no-deps -vv
```

**Step 3: 创建 bld.bat**

```bat
@echo off
python install_flash_attn_stub.py
if errorlevel 1 exit 1
%PYTHON% -m pip install . --no-deps -vv
if errorlevel 1 exit 1
```

**Step 4: Commit**

```bash
git add conda.recipe/build.sh conda.recipe/bld.bat install_flash_attn_stub.py
git commit -m "feat: add conda build scripts and flash_attn stub installer"
```

---

## Task 9: 验证测试

**Step 1: 手动安装依赖并测试导入**

```bash
# 先安装 flash_attn stub
python install_flash_attn_stub.py

# 然后测试导入链
cd /media/asky/F/Euktect_edit/Euktect
python -c "
import sys
sys.path.insert(0, 'hyena-dna')
from src.models.sequence.long_conv_lm import DNAEmbeddingModel
from src.tasks.decoders import SequenceDecoder
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
print('All imports successful!')
"
```

期望输出: `All imports successful!`

**Step 2: 测试 predict.py 基本参数解析**

```bash
python predict.py --help
```

期望: 打印帮助文档，无 CUDA 相关错误

**Step 3: 验证无 GPU 依赖**

```bash
python -c "import flash_attn; print(flash_attn.__file__)"
# 期望输出包含 "flash_attn" 路径（来自 stub）

python -c "
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp, FusedMLP
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.distributed import sync_shared_params
print('All flash_attn stubs import OK')
"
```

**Step 4: 如测试通过，Commit 测试结果**

```bash
git commit --allow-empty -m "test: validate CPU-only imports successfully"
```

---

## Task 10: 最终检查和 README 更新

**Step 1: 最终文件清单验证**

```
Euktect/
├── flash_attn_cpu/          ← flash_attn CPU stub
│   ├── __init__.py
│   ├── modules/{mha,mlp,block,embedding}.py
│   ├── utils/{generation,distributed}.py
│   ├── ops/{fused_dense,layer_norm}.py
│   └── losses/cross_entropy.py
├── conda.recipe/
│   ├── meta.yaml
│   ├── build.sh
│   └── bld.bat
├── install_flash_attn_stub.py
├── setup.py
├── euktect/__init__.py
├── predict.py               ← 无需修改
├── refine.py               ← 无需修改
└── hyena-dna/
    └── src/models/sequence/block_fft.py  ← 已修复 .cuda()
```

**Step 2: 检查 block_fft.py 修改是否完整**

```bash
grep -n "cuda" /media/asky/F/Euktect_edit/Euktect/hyena-dna/src/models/sequence/block_fft.py
```

**Step 3: Final commit**

```bash
git add .
git commit -m "feat: complete CPU-only conda package for Euktect"
```

---

## 构建命令（供用户使用）

完成上述任务后，用户可以使用以下命令构建 conda 包：

```bash
# 安装 conda-build
conda install conda-build

# 构建包
conda build conda.recipe/ -c pytorch -c conda-forge --no-test

# 安装构建好的包
conda install --use-local euktect
```

或者不使用 conda-build 直接在已有环境中安装：

```bash
# 安装 CPU-only PyTorch（如尚未安装）
conda install pytorch cpuonly -c pytorch

# 安装其他依赖
pip install pytorch-lightning==1.8.6 transformers==4.26.1 hydra-core omegaconf einops timm pyfaidx polars loguru

# 安装 flash_attn CPU stub
python install_flash_attn_stub.py

# 安装 euktect
pip install -e .
```
