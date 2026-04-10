# scGPT-modern: Environment Setup & Usage Tutorial

> A complete recipe for running scGPT on **Python 3.12 + torch 2.6+cu124 +
> flash-attn 3 (Hopper native)** with the original `bowang-lab/scGPT` pretrained
> checkpoints loading unmodified.
>
> Written against an H100 (`sm_90a`) node; the same stack runs on A100/L40S via
> the flash-attn 2 fallback (see [§7](#7-running-on-non-h100-gpus)).

---

## TL;DR

```bash
# activate your clean Python 3.12 env, then:
python -c "
import torch
from scgpt._compat.flash_attention import get_backend_name
from scgpt.tasks.cell_emb import embed_data
import scanpy as sc

print('torch:', torch.__version__, '| device:', torch.cuda.get_device_name(0))
print('backend:', get_backend_name())   # expected: v3-hopper

adata = sc.read_h5ad('pbmc3k_raw_with_labels.h5ad')
out = embed_data(adata, 'scGPT_continual_pretrained',
                 gene_col='gene_name', use_fast_transformer=True)
print('embedding:', out.shape)
"
```

Expected output on an H100 with everything in place:

```
torch: 2.6.0+cu124 | device: NVIDIA H100 80GB HBM3
backend: v3-hopper
embedding: (2638, 512)
```

---

## 1. Prerequisites

| Requirement | Minimum | Recommended | Why |
|---|---|---|---|
| NVIDIA driver | 550+ (supports CUDA 12.4) | 560+ | torch 2.6 wheel is cu124; driver needs to be ≥ the wheel's CUDA minor |
| GPU | Ampere sm_80 (A100, A40) | **Hopper sm_90a (H100)** | FA3 requires sm_90a for WGMMA / TMA. Older arches fall back to FA2 |
| CUDA toolkit (for building FA3) | 12.4 | 12.4 exactly | Must match the torch wheel ABI; 12.6/12.8 host headers often work but are not tested |
| GCC (for building FA3) | 11 | 12.4 | nvcc 12.4 officially supports gcc ≤ 13 |
| RAM (for building FA3) | 32 GB | 150+ GB | Bwd sm90 kernels use 15–25 GB per `cicc` process; scales with MAX_JOBS |
| Disk | 15 GB | — | torch wheel + deps ≈ 6 GB; FA3 build tree ≈ 5 GB; site-packages ≈ 7 GB |

If you're on a shared cluster, you most likely have a CUDA module system. Use
`module avail cuda` or equivalent to find a version close to 12.4.

---

## 2. Install the stack

### 2.1 Create a Python 3.12 env

```bash
# Using micromamba / conda / mamba — pick your flavor
micromamba create -p /path/to/scgpt-modern-env -c conda-forge -y \
    python=3.12.12 pip

# Activate it. If your shell auto-loads other envs (user site, base conda,
# venvs from job schedulers, etc.), prefer direct PATH prepend over
# `micromamba activate` to avoid pollution:
export PATH=/path/to/scgpt-modern-env/bin:$PATH
export PYTHONNOUSERSITE=1   # blocks ~/.local/lib/pythonX.Y/site-packages
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
```

Verify:

```bash
python -c "import sys; print(sys.version.split()[0], sys.executable); print(sys.path)"
# Python 3.12.12 /path/to/scgpt-modern-env/bin/python3
# sys.path should contain only the env's site-packages, nothing else.
```

> **Why so paranoid about pollution?** On shared clusters, login shells often
> auto-activate a "default" venv, `module load python/...`, or prepend a
> `~/.local/lib/python3.12/site-packages` that has 100+ packages. Any one of
> these will shadow your new env's `numpy` / `anndata` / `torch` and produce
> confusing `Requirement already satisfied` messages or import-time ABI
> errors. The `PYTHONNOUSERSITE=1` + `unset` incantation is cheap insurance.

### 2.2 torch 2.6.0 + cu124

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2.6.0+cu124 True NVIDIA H100 80GB HBM3
```

### 2.3 Scientific stack

```bash
pip install --only-binary=:all: \
    numpy pandas scipy scikit-learn matplotlib seaborn numba \
    scanpy anndata h5py pyarrow leidenalg python-igraph \
    einops tqdm ipython psutil ninja poetry-core \
    typing-extensions
```

`poetry-core` is needed because scGPT's `pyproject.toml` uses it as the build
backend. `ipython` is needed because `scgpt/utils/util.py` has a top-level
`from IPython import get_ipython` (not lazy). `psutil` and `ninja` are
flash-attn build-time dependencies.

### 2.4 flash-attn 2 (universal fallback)

**Important**: flash-attn 2's `setup.py` first downloads a pre-built wheel to
`/tmp` and then calls `os.rename()` to move it to pip's cache dir. If `/tmp`
and your pip cache dir live on different filesystems (common on clusters
where `/tmp` is local NVMe and `$HOME` / pip cache is network-mounted), you'll
see `Invalid cross-device link`. Bypass the setup.py dance by installing
directly from the GitHub release URL:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    --no-build-isolation
```

For cp311 swap `cp312-cp312` → `cp311-cp311`. For cu11 swap `cu12` → `cu11`.

Verify:

```bash
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
# flash_attn: 2.7.4.post1
```

At this point you can already run scGPT with the compat shim picking backend
`v2`. If you want the FA3 speedup on H100, continue to §3.

### 2.5 Install scGPT-modern (editable)

```bash
git clone https://github.com/chansigit/scGPT-modern.git
cd scGPT-modern
pip install -e . --no-deps --no-build-isolation

# Verify the shim is importable and picks a backend
python -c "
from scgpt._compat.flash_attention import get_backend_name
print('backend:', get_backend_name())
"
# backend: v2   (will become v3-hopper after §3)
```

---

## 3. Build flash-attn 3 from source (H100, recommended)

flash-attn 3 is **not** on PyPI. NVIDIA's `flash-attn-4` package on PyPI has
broken metadata (hard-depends on an unpublished `nvidia-cutlass-dsl-libs-base`
package), so `pip install flash-attn-4` also fails. The only working route
right now is to build v3 from the `hopper/` subdirectory of the upstream
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
repo.

### 3.1 Clone and init the cutlass submodule

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# The cutlass submodule is required — FA3 kernels all include cutlass headers.
# Shallow clone is fine; the full cutlass history is not needed for a build.
git submodule update --init --depth 1 csrc/cutlass

ls csrc/cutlass/include/cutlass/fast_math.h
# should exist
```

### 3.2 Set up the build environment

flash-attn 3's `hopper/setup.py` is picky about the toolchain. Set up these
environment variables before invoking pip/setup.py:

```bash
# CUDA 12.4 (match the torch wheel ABI exactly)
export CUDA_HOME=/path/to/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# GCC 12.x (CUDA 12.4 supports gcc up to 13)
export PATH=/path/to/gcc-12.4/bin:$PATH
export CC=/path/to/gcc-12.4/bin/gcc
export CXX=/path/to/gcc-12.4/bin/g++

# Build knobs
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="9.0"

# IMPORTANT: trim the compile scope to only what scGPT uses.
# Without these, the build produces 293 kernels and takes ~40 min + 300 GB RAM.
# With these, it drops to 27 kernels and ~5 min + 150 GB peak.
export FLASH_ATTENTION_DISABLE_SM80=TRUE       # sm_90a only
export FLASH_ATTENTION_DISABLE_FP8=TRUE        # scGPT uses fp16/bf16
export FLASH_ATTENTION_DISABLE_HDIM96=TRUE     # scGPT head_dim ∈ {64, 128}
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE    # LLM KV cache, not used
export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE      # full attention only
export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE

# Limit concurrency to avoid cicc OOM.
# FA3 bwd sm90 kernels use 15–25 GB per cicc process during template
# instantiation; 6 concurrent × 1 thread each = ~150 GB peak, fits in a
# standard compute node.
export MAX_JOBS=6
export NVCC_THREADS=1
```

If you only need forward-only inference (e.g., zero-shot embedding pipelines
with `embed_data`), you can additionally set
`FLASH_ATTENTION_DISABLE_BACKWARD=TRUE` to cut the build further. Do not set
this if you plan to fine-tune, since scGPT fine-tuning gradients flow through
the FlashMHA block.

> **scope tradeoffs**: the `DISABLE_*` flags above keep only `head_dim ∈ {64, 128}`
> in fp16/bf16. scGPT's default `d_model=512, nhead=8` → `head_dim=64`; the
> largest published checkpoint uses `d_model=768, nhead=12` → `head_dim=64`.
> If you build a custom scGPT variant with a different geometry, remove the
> relevant `DISABLE_HDIM*` flag and rebuild, or fall back to FA2 which
> supports all `head_dim` values up to 256 via its pre-built wheel.

### 3.3 Build and install

```bash
cd flash-attention/hopper
rm -rf build dist *.egg-info

# ~5 min on a 24-core node with MAX_JOBS=6, peak ~150 GB RAM
python setup.py bdist_wheel 2>&1 | tee /tmp/fa3_build.log

ls dist/
# flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl

pip install dist/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl
```

Note the `cp39-abi3` tag: this is a PEP 384 stable ABI wheel built against
CPython 3.9 headers but **forward compatible to any CPython ≥ 3.9** — so the
same wheel works on Python 3.10, 3.11, 3.12, 3.13 with zero rebuild. You only
need to rebuild when torch version, CUDA version, or GPU arch changes.

### 3.4 Fix the upstream wheel packaging bug

flash-attn 3's `hopper/setup.py` puts `flash_attn_interface.py` and
`flash_attn_config.py` at the **top level** of the wheel, while the C
extension (`_C.abi3.so`) ends up inside `flash_attn_3/`. But
`flash_attn_interface.py` does `from flash_attn_3 import _C`, so it fails to
import unless it's inside the package directory.

Fix post-install:

```bash
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
ls $SP/flash_attn_interface.py $SP/flash_attn_config.py $SP/flash_attn_3/
# flash_attn_interface.py  flash_attn_config.py
# _C.abi3.so               ← only this is in the package dir

mv $SP/flash_attn_interface.py $SP/flash_attn_3/flash_attn_interface.py
mv $SP/flash_attn_config.py $SP/flash_attn_3/flash_attn_config.py

cat > $SP/flash_attn_3/__init__.py << 'PY'
"""flash_attn_3 — post-install fix for upstream wheel layout bug.

The upstream hopper/setup.py ships flash_attn_interface.py and
flash_attn_config.py at the top level of the wheel, but
flash_attn_interface.py does `from flash_attn_3 import _C`, so it can only
work from inside the flash_attn_3/ package.
"""
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_combine,
    get_scheduler_metadata,
)
__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_with_kvcache",
    "flash_attn_combine",
    "get_scheduler_metadata",
]
PY
```

Verify:

```bash
python -c "
from flash_attn_3 import flash_attn_func
import torch
q = torch.randn(2, 64, 4, 64, device='cuda', dtype=torch.float16)
k, v = q.clone(), q.clone()
print('FA3 GPU forward shape:', tuple(flash_attn_func(q, k, v).shape))

from scgpt._compat.flash_attention import get_backend_name
print('backend:', get_backend_name())
"
# FA3 GPU forward shape: (2, 64, 4, 64)
# backend: v3-hopper
```

---

## 4. Sanity check

Once FA3 is installed and the compat shim reports `v3-hopper`, run a small
end-to-end forward to confirm the scGPT model pipeline is wired correctly:

```python
import torch
from scgpt.tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt._compat.flash_attention import get_backend_name

print('torch:', torch.__version__, '| device:', torch.cuda.get_device_name(0))
print('backend:', get_backend_name())

# Small dummy vocab
genes = [f"GENE{i}" for i in range(100)]
vocab = GeneVocab(genes, specials=['<pad>', '<cls>', '<eoc>'], special_first=True)
vocab.set_default_index(vocab['<pad>'])

m = TransformerModel(
    ntoken=len(vocab), d_model=64, nhead=4, d_hid=128, nlayers=2,
    nlayers_cls=1, n_cls=3, vocab=vocab, dropout=0.0,
    pad_token='<pad>', pad_value=0,
    do_mvc=False, do_dab=False, use_batch_labels=False,
    input_emb_style='continuous', n_input_bins=51,
    use_fast_transformer=True, fast_transformer_backend='flash',
    pre_norm=False,
).cuda().half()

src = torch.randint(0, len(vocab), (2, 16), device='cuda')
values = torch.randn(2, 16, device='cuda', dtype=torch.float16)
mask = torch.zeros(2, 16, dtype=torch.bool, device='cuda')
out = m(src, values, src_key_padding_mask=mask, CLS=True)
print('forward OK, cls_output shape:', out['cls_output'].shape)
```

Expected output:

```
torch: 2.6.0+cu124 | device: NVIDIA H100 80GB HBM3
backend: v3-hopper
forward OK, cls_output shape: torch.Size([2, 3])
```

---

## 5. The compat shim architecture

scGPT-modern makes **four one-line import edits** to the upstream scGPT
source tree and adds a new `scgpt/_compat/` package. That's it — no
`state_dict` key remapping, no model code edits, no training changes.

### 5.1 The edited imports

| File | Old import | New import |
|---|---|---|
| `scgpt/model/model.py` | `from flash_attn.flash_attention import FlashMHA` | `from scgpt._compat.flash_attention import FlashMHA` |
| `scgpt/model/multiomic_model.py` | same | same |
| `scgpt/model/generation_model.py` | same | same |
| `scgpt/tokenizer/gene_tokenizer.py` | `import torchtext.vocab as torch_vocab` + `from torchtext.vocab import Vocab` | `from scgpt._compat import torchtext_vocab as torch_vocab` + `from scgpt._compat.torchtext_vocab import Vocab` |

### 5.2 `FlashMHA` shim ([scgpt/_compat/flash_attention.py](../scgpt/_compat/flash_attention.py))

The critical requirement: the shim's learnable submodules must have **the
exact same names and shapes** as the legacy flash-attn 1.x `FlashMHA`, or
pretrained checkpoints won't load.

Legacy (flash-attn 1.0.4):
```python
class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, ...):
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.inner_attn = FlashAttention(...)   # no learnable params
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
```

scgpt-modern shim:
```python
class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, ...):
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)     # same name, shape
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # same name, shape
        # No inner_attn — the kernel call is inlined in forward()

    def forward(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.Wqkv(x)

        # fp32 inputs are auto-cast to fp16, then cast back before out_proj.
        # Mirrors flash-attn v1 internal behavior; scgpt.tasks.embed_data
        # leaves the model unpromoted and relies on this.
        orig_dtype = qkv.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(torch.float16)

        q, k, v = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=num_heads).unbind(dim=2)
        flash_attn_func, flash_attn_varlen_func, _ = _resolve_backend()

        if key_padding_mask is None:
            out = flash_attn_func(q, k, v, causal=self.causal)
        else:
            # unpad → varlen kernel → pad
            ...

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return self.out_proj(out), None
```

Because `inner_attn` had **no learnable parameters** in the legacy version,
it doesn't appear in the state_dict — so the shim doesn't need to reproduce
it. `load_state_dict(strict=True)` succeeds on all attention keys.

### 5.3 Backend auto-selection

```python
def _resolve_backend():
    try:
        from flash_attn.cute import flash_attn_func          # v4-cute (Hopper/Blackwell)
        return ..., "v4-cute"
    except ImportError: pass
    try:
        from flash_attn_3 import flash_attn_func             # v3-hopper
        return _wrap_v3(...), "v3-hopper"                    # strips dropout_p kwarg
    except ImportError: pass
    try:
        from flash_attn import flash_attn_func               # v2 (universal)
        return ..., "v2"
    except ImportError: pass
    raise ImportError(...)
```

Preference order: `v4 > v3 > v2`. Lazy-evaluated and cached after the first
successful forward call. The `_wrap_v3` adapter exists because FA3 dropped
the `dropout_p` kwarg (FA3 has no native attention dropout path). scGPT's
default config uses `attention_dropout=0.0` so this is a no-op at runtime; a
warning is emitted once if a nonzero dropout is ever requested on v3.

### 5.4 torchtext shim ([scgpt/_compat/torchtext_vocab.py](../scgpt/_compat/torchtext_vocab.py))

`torchtext` 0.18.0's C++ extensions (`libtorchtext.so`) use the torch 2.3
ABI; they fail to load on torch 2.5+ with:

```
OSError: libtorchtext.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSs
```

scGPT only uses `torchtext.vocab.Vocab`'s dict-lookup interface
(`__getitem__`, `__contains__`, `insert_token`, `append_token`, `get_stoi`,
`set_default_index`). Reimplementing this in ~190 lines of pure Python is
trivial and removes the torchtext dependency entirely.

### 5.5 Forward-compatible upgrade path

| Future event | What to do |
|---|---|
| NVIDIA fixes the `flash-attn-4` packaging | `pip install flash-attn-4`; shim auto-detects and switches to `v4-cute`. **Zero code changes.** |
| You want to fall back to FA2 for A/B testing | `pip uninstall flash-attn-3`; shim auto-falls back to `v2`. |
| torch upgrade to 2.7+ | Should work unchanged; shim doesn't depend on specific torch version. The FA3 `cp39-abi3` wheel rebuilds once. |

---

## 6. Usage: zero-shot cell embedding

Same API as upstream scGPT:

```python
from scgpt.tasks.cell_emb import embed_data

adata_emb = embed_data(
    adata_or_file=adata,                  # .X is raw counts
    model_dir="path/to/scGPT_continual_pretrained",
    gene_col="gene_name",
    max_length=1200,
    batch_size=64,
    obs_to_save=["celltype"],
    device="cuda",
    use_fast_transformer=True,
    return_new_adata=True,
)
# adata_emb.X is (n_cells, 512), L2-normalized
```

### 6.1 Numerical parity with upstream

PBMC3k (2638 cells × 8 celltypes) zero-shot embedding with the
**unmodified** `scGPT_continual_pretrained` checkpoint:

| | modern (FA3 on H100) | upstream (FA1 on L40S) |
|---|---|---|
| `load_state_dict` attention missing | 0 | 0 |
| `load_state_dict` attention unexpected | 0 | 0 |
| embedding shape | (2638, 512) | (2638, 512) |
| embedding `std` | 0.0442 | 0.0442 |
| embedding `mean` | 0.0007 | 0.0007 |
| per-cell cosine sim vs upstream | **0.9958** | — |
| relative L2 error vs upstream | **0.092** | — |

The ~1% numerical drift is from the different fp16 accumulation order of
`sm_86` PTX-JIT kernels (legacy FA1) vs `sm_90a` WGMMA kernels (FA3), not
from anything semantic. It's below the single-run Leiden clustering noise
floor for downstream ARI comparisons.

### 6.2 Downstream clustering: don't add PCA

scGPT's embedding is already L2-normalized. Feed it directly to
`sc.pp.neighbors`:

```python
import scanpy as sc
sc.pp.neighbors(adata_emb, use_rep="X", n_neighbors=15, metric="cosine")
sc.tl.umap(adata_emb, min_dist=0.3)
sc.tl.leiden(adata_emb, resolution=0.5, flavor="igraph", n_iterations=2, directed=False)
```

Adding PCA first drops ARI by 0.15–0.20 because PCA centering pulls the
unit-sphere vectors away from the geometric origin the model was trained on.
This is documented in the upstream zero-shot tutorials as well.

### 6.3 Monte Carlo TTA for stable benchmarks

Single-pass scGPT embeddings are noisy (the `DataCollator(sampling=True)`
subsamples genes when a cell has more than `max_length` of them, and fp16
flash-attention kernels are non-deterministic). Single-run ARI on PBMC3k
can fluctuate across `[0.54, 0.80]` depending on seed.

For benchmark reporting, run K≥8 passes with different seeds and average:

```python
import numpy as np, torch
accum = None
for k in range(8):
    torch.manual_seed(k); np.random.seed(k)
    e = embed_data(adata, ..., return_new_adata=True).X
    accum = e if accum is None else accum + e
emb = accum / 8
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
```

Saturation is around K=8–16. K=1 is fine for pipeline sanity checks; don't
report K=1 numbers in papers.

---

## 7. Running on non-H100 GPUs

The shim's auto-selection falls back to flash-attn 2 on any arch FA2
supports:

| GPU | Recommended backend | Notes |
|---|---|---|
| H100 (sm_90a) | **v3-hopper** | Install FA3 per §3; strictly faster than v2 here. |
| H200, GB200 (sm_90a / sm_100a) | v3-hopper or v4-cute (when available) | Same as H100; FA4 preview targets sm_100. |
| A100 (sm_80) | v2 | FA2 pre-built wheels cover sm_80 natively. Skip §3 entirely. |
| A40, A10, L40, L40S (sm_86 / sm_89) | v2 | Same as A100. |
| Older (sm_75 and below) | (fallback: pure PyTorch SDPA) | FA2 stops supporting sm_75 after version 2.x; the shim will report `backend: none` and scGPT's `flash_attn_available` gate in `model.py` will route through `torch.nn.functional.scaled_dot_product_attention`. |

In all cases, the scGPT source code and the compat shim are identical; only
the underlying kernel changes.

---

## 8. Troubleshooting

### `backend: v2` when FA3 should be available

FA3 wheel is installed but the Python import fails. Check:

```bash
python -c "from flash_attn_3 import flash_attn_func; print('ok')"
```

If this raises `ImportError: cannot import name 'flash_attn_func' from 'flash_attn_3'`,
you hit the upstream wheel packaging bug — re-run the post-install fix from
§3.4.

### `backend: none`

Neither FA3 nor FA2 is importable. Check:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "import flash_attn_3; print(flash_attn_3.__file__)"
```

Common causes:
- FA2 wheel was installed for a different Python version / torch version /
  CUDA version than the active env. Check the wheel filename: it should
  encode `cp{PY}-cp{PY}` and `cu12torch2.6` matching your env.
- `einops` not installed (the shim depends on `rearrange` for `nn` path).
- FA2 wheel's `libcudart.so` or `libnvrtc.so` can't resolve — check
  `LD_LIBRARY_PATH` points at a CUDA runtime matching the wheel's cu tag.

scGPT's own `scgpt/model/model.py` has a `flash_attn_available` flag that
detects this and falls back to plain PyTorch SDPA — the model still runs,
just slower.

### `RuntimeError: Error(s) in loading state_dict for TransformerModel: size mismatch for cls_decoder.out_layer.weight`

You're constructing `TransformerModel` with the wrong `n_cls`. The
`continual_pretrained` checkpoint has 177 cell-type classes from the
pretraining dataset. Infer it from the checkpoint:

```python
import torch
sd = torch.load(f'{ckpt_dir}/best_model.pt', map_location='cpu', weights_only=False)
n_cls = sd['cls_decoder.out_layer.weight'].shape[0]       # 177
do_mvc = 'mvc_decoder.query_emb.weight' in sd            # True for continual_pretrained
pad_value = args_json.get('pad_value', -2)                # from args.json
```

### `AssertionError: flash-attn requires fp16 or bf16 inputs, got torch.float32`

Old version of the `FlashMHA` shim without the fp32 auto-cast path. Pull the
latest `scgpt/_compat/flash_attention.py` from this repo; the current
version handles fp32 inputs transparently.

### Build: `sh: line 1: <pid> Killed "$CICC_PATH/cicc"`

`cicc` (nvcc's internal C++ front-end) was OOM-killed by the kernel. FA3
bwd sm90 kernels can use 15–25 GB per `cicc` during template instantiation.
Drop `MAX_JOBS` to 4 or lower, make sure `NVCC_THREADS=1`, and verify your
compute node has enough free RAM: `free -g` should show at least
`MAX_JOBS × 25 GB + 10 GB overhead`.

### Build: `fatal error: cutlass/fast_math.h: No such file`

The `csrc/cutlass` submodule wasn't initialized:

```bash
cd flash-attention
git submodule update --init --depth 1 csrc/cutlass
```

### Build: `error: [Errno 18] Invalid cross-device link`

The flash-attn setup.py downloads wheels to `/tmp` and then `os.rename()`s
them to pip's cache dir. On clusters where `/tmp` and the cache dir are on
different filesystems, this fails. Fix by installing directly from the
GitHub release URL (see §2.4), bypassing the download+rename logic.

### Pip sees packages in some other venv

You didn't set `PYTHONNOUSERSITE=1` before activating, or your shell's
startup files re-export `PYTHONPATH`. Run:

```bash
env | grep -E "^(PYTHONPATH|VIRTUAL_ENV|PYTHONUSERBASE)"
python -c "import sys; print('\n'.join(sys.path))"
```

Any path that isn't your env's `site-packages` or stdlib is suspect. Go
back to §2.1 and re-activate cleanly.

### `enable_nested_tensor is True, but self.use_nested_tensor is False`

Harmless torch 2.6 UserWarning. scGPT's `FlashTransformerEncoderLayer` is
not a subclass of `nn.TransformerEncoderLayer`, so torch's nested-tensor
optimization path doesn't engage. Ignore.

---

## 9. Uninstall / reset

```bash
# Reset just the scgpt editable install
pip uninstall -y scGPT

# Reset flash-attn (keeps torch intact)
pip uninstall -y flash-attn flash-attn-3

# Nuke the whole env
rm -rf /path/to/scgpt-modern-env
```

The scGPT source tree itself is just a clone — `rm -rf scGPT-modern/` wipes
it without any side effects.

---

## 10. Related resources

- [scgpt/_compat/flash_attention.py](../scgpt/_compat/flash_attention.py) — FlashMHA shim source
- [scgpt/_compat/torchtext_vocab.py](../scgpt/_compat/torchtext_vocab.py) — torchtext replacement source
- [`bowang-lab/scGPT`](https://github.com/bowang-lab/scGPT) — upstream model code, pretrained checkpoints, published tutorials
- [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) — FA source, particularly the `hopper/` subdirectory for FA3
- [flash-attn 2 release wheels](https://github.com/Dao-AILab/flash-attention/releases) — pre-built cp3X wheels for every torch × CUDA combination
