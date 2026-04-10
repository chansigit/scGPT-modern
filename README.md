# scGPT-modern

> A **drop-in modernization** of [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
> for **Python 3.12 + torch 2.6 + flash-attn 3** (H100 `sm_90a` native).
> **The original pretrained checkpoints load unmodified** — this is purely a
> runtime / dependency upgrade, not a new training run.

## What's different from upstream

| | upstream `bowang-lab/scGPT` | this repo `chansigit/scGPT-modern` |
|---|---|---|
| Python | 3.7 – 3.10 | **3.12** |
| torch | 1.13.x + cu117 | **2.6.0 + cu124** |
| numpy | locked `< 2` | **≥ 2** (free) |
| flash-attn | v1.0.4 (patched for sm_86 PTX) | **v3.0.0 (hopper)** native `sm_90a`, auto-falls back to v2 |
| torchtext dependency | required (0.14 for torch 1.13) | **removed** (pure-Python shim) |
| Target GPU | A100 / A40 / L40S | **H100** (primary), still runs on older arch via the v2 fallback |
| scGPT source changes | — | **4 import-line edits** (see [`scgpt/_compat/`](scgpt/_compat/)) |

## Why this exists

The upstream code has a hard `from flash_attn.flash_attention import FlashMHA`
import pinned to the legacy flash-attn 1.x API, and an `import torchtext.vocab`
that breaks under torch 2.5+ (ABI mismatch — `libtorchtext.so` fails to load
with undefined-symbol errors).

Rather than fork the model code, this repo adds a thin `scgpt/_compat/` shim
package that:

1. Reproduces the old `FlashMHA` class with **the exact same submodule names**
   (`Wqkv`, `out_proj`) so pretrained `state_dict`s load with **zero key
   remapping**.
2. Provides a ~190-line pure-Python replacement for `torchtext.vocab.Vocab` —
   scGPT only touches 5 methods of it, no reason to keep the C++ dependency.

The shim auto-selects the best available attention backend at runtime:

```
flash-attn-4 (CuTeDSL, when NVIDIA ships working packaging)
  ↓ fallback
flash-attn-3 (hopper, sm_90a native — source-built from Dao-AILab/flash-attention hopper/)
  ↓ fallback
flash-attn-2 (pre-built wheel, covers Ampere / Ada / Hopper)
  ↓ fallback
pure PyTorch SDPA (if none of the above import — via the upstream `flash_attn_available` gate)
```

FA3 removed the `dropout_p` kwarg, so the shim wraps it in a tiny adapter.
fp32 inputs are auto-cast to fp16 inside `FlashMHA.forward()` and cast back
before `out_proj` — this mirrors flash-attn v1's internal behavior, which the
`scgpt.tasks.embed_data` pipeline relies on (it leaves the model unpromoted).

## Verified

PBMC3k zero-shot cell embedding with the **unmodified** `scGPT_continual_pretrained`
checkpoint (208 MB) produces results **numerically near-identical** to the
legacy flash-attn-1.0.4 run on L40S:

| | this fork (FA3 on H100) | upstream env (FA1 on L40S) |
|---|---|---|
| `load_state_dict` missing keys on attention layers | **0** | 0 |
| `load_state_dict` unexpected keys on attention layers | **0** | 0 |
| embedding shape | (2638, 512) | (2638, 512) |
| embedding `std` | 0.0442 | 0.0442 |
| embedding `mean` | 0.0007 | 0.0007 |
| per-cell cosine similarity vs legacy run | **0.9958** | — |
| relative L2 error vs legacy run | **0.092** | — |

The tiny numerical drift comes from sm_86 PTX-JIT vs sm_90a WGMMA fp16
accumulation order — it's well below the single-run Leiden clustering noise
floor (ARI ± 0.05 across random seeds) documented in the upstream zero-shot
tutorials.

## Quick start

> See [`docs/scGPT-modern-env-tutorial.md`](docs/scGPT-modern-env-tutorial.md)
> for the full setup guide: prerequisites, cluster pollution pitfalls, FA3
> build from source with trimmed scope, the compat shim architecture deep-dive,
> MC-TTA for stable benchmarks, per-GPU backend matrix, and a troubleshooting
> section for the failure modes we actually hit.

```bash
# 1. Create a Python 3.12 env (micromamba / conda / venv — your pick)
micromamba create -p /path/to/env -c conda-forge -y python=3.12 pip

# 2. Install torch 2.6 + cu124 from pytorch.org
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3. Install flash-attn 2 as a universal fallback (cp312 wheel from GitHub release)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl \
    --no-build-isolation

# 4. (Recommended on H100) build flash-attn 3 from source — ~5 min with trimmed
#    scope (sm_90a only; disable fp8 / hdim 96 / 192 / 256 / paged / local / cluster).
#    See https://github.com/Dao-AILab/flash-attention/tree/main/hopper
#
#    The resulting wheel is tagged cp39-abi3 and is forward-compatible to any
#    CPython >= 3.9, so you only ever build it once even across Python upgrades.

# 5. Scientific stack (all binary wheels)
pip install --only-binary=:all: \
    numpy scanpy anndata pandas scipy scikit-learn numba \
    einops ipython leidenalg python-igraph h5py pyarrow

# 6. scGPT-modern itself (editable)
pip install poetry-core
git clone https://github.com/chansigit/scGPT-modern.git
cd scGPT-modern
pip install -e . --no-deps --no-build-isolation

# 7. Verify the backend resolver picks the fastest available kernel
python -c "
from scgpt._compat.flash_attention import get_backend_name
print('FlashMHA backend:', get_backend_name())
# expected on H100 with FA3 installed:      v3-hopper
# expected on H100 with only FA2 installed: v2
# expected with nothing importable:         none
"
```

## Usage

**Identical to upstream.** The compat shim is completely transparent once the 4
import edits are in place. Every existing scGPT workflow — `embed_data`,
`get_batch_cell_embeddings`, `TransformerModel`, `MultiomicModel`, the
fine-tuning notebooks, the zero-shot tutorials — runs unchanged.

```python
from scgpt.tasks.cell_emb import embed_data
import scanpy as sc

adata = sc.read_h5ad("path/to/raw_counts.h5ad")
out = embed_data(
    adata,
    model_dir="path/to/scGPT_continual_pretrained",
    gene_col="gene_name",
    use_fast_transformer=True,   # this triggers the shim backend resolver
    return_new_adata=True,
)
# out.X is (n_cells, 512), L2-normalized, ready for sc.pp.neighbors(metric="cosine")
```

## Relationship to upstream

This repo is a **standalone upgrade**, not a GitHub fork. The commit history
below the two modernization commits is identical to upstream `bowang-lab/scGPT`
at the branch point; an `upstream` git remote can be added locally for pulling
future upstream changes:

```bash
git remote add upstream https://github.com/bowang-lab/scGPT.git
git fetch upstream && git merge upstream/main
```

**Where to file issues:**
- Questions about the scGPT model itself, training, checkpoints, published
  results → [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
- Issues specific to the modernized stack (`_compat/` shim, torch 2.x build,
  flash-attn 3, H100 deployment) → this repo

## License

Same as upstream scGPT (MIT). See [`LICENSE`](LICENSE).

---

# Original scGPT README

_Everything below this line is preserved verbatim from the upstream project._

This is the official codebase for **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI**.

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) &nbsp;
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;
[![PyPI version](https://badge.fury.io/py/scgpt.svg)](https://badge.fury.io/py/scgpt) &nbsp;
[![Downloads](https://pepy.tech/badge/scgpt)](https://pepy.tech/project/scgpt) &nbsp;
![Webapp](https://img.shields.io/website?url=https%3A%2F%2Fscgpthub.org&up_color=chartreuse%20&logo=gotomeeting&logoColor=%23FFB3FF&label=WebApp&labelColor=%2300CBFF) &nbsp;
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

**!UPDATE**: We have released several new pretrained scGPT checkpoints. Please see the [Pretrained scGPT checkpoints](#pretrained-scGPT-checkpoints) section for more details.

**[2024.02.26]** We have provided a priliminary support for running the pretraining workflow with HuggingFace at the [integrate-huggingface-model](https://github.com/bowang-lab/scGPT/tree/integrate-huggingface-model) branch. We will conduct further testing and merge it to the main branch soon.

**[2023.12.31]** New tutorials about zero-shot applications are now available! Please see find them in the [tutorials/zero-shot](tutorials/zero-shot) directory. We also provide a new continual pretrained model checkpoint for cell embedding related tasks. Please see the [notebook](tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb) for more details.

**[2023.11.07]** As requested by many, now we have made flash-attention an optional dependency. The pretrained weights can be loaded on pytorch CPU, GPU, and flash-attn backends using the same [load_pretrained](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/utils/util.py#L304) function, `load_pretrained(target_model, torch.load("path_to_ckpt.pt"))`. An example usage is also [here](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/tasks/cell_emb.py#L258).

**[2023.09.05]** We have release a new feature for reference mapping samples to a custom reference dataset or to all the millions of cells collected from CellXGene! With the help of the [faiss](https://github.com/facebookresearch/faiss) library, we achieved a great time and memory efficiency. The index of over 33 millions cells only takes less than 1GB of memory and the similarity search takes less than **1 second for 10,000 query cells on GPU**. Please see the [Reference mapping tutorial](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Reference_Mapping.ipynb) for more details.

### Online apps

scGPT is now available at the following online apps as well, so you can get started simply with your browser!

- Run the [reference mapping app](https://app.superbio.ai/apps/299?id=6548f339a9ed6f6e5560b07d), [cell annotation app](https://app.superbio.ai/apps/274?id=64d205cb980ff714de831ee0) and the [GRN inference app](https://app.superbio.ai/apps/270?id=64b804fb823bc93b64c10a76) with cloud gpus. Thanks to the [Superbio.ai](https://app.superbio.ai/) team for helping create and host the interactive tools.

## Installation

scGPT works with Python >= 3.7.13 and R >=3.6.1. Please make sure you have the correct version of Python and R installed pre-installation.

scGPT is available on PyPI. To install scGPT, run the following command:

```bash
pip install scgpt "flash-attn<1.0.5"  # optional, recommended
# As of 2023.09, pip install may not run with new versions of the google orbax package, if you encounter related issues, please use the following command instead:
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

[Optional] We recommend using [wandb](https://wandb.ai/) for logging and visualization.

```bash
pip install wandb
```

The poetry installation is out of sync. Please use pip install instead. ~~For developing, we are using the [Poetry](https://python-poetry.org/) package manager. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).~~

```bash
$ git clone this-repo-url
$ cd scGPT
$ poetry install
```

**Note**: The `flash-attn` dependency usually requires specific GPU and CUDA version. If you encounter any issues, please refer to the [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) repository for installation instructions. For now, May 2023, we recommend using CUDA 11.7 and flash-attn<1.0.5 due to various issues reported about installing new versions of flash-attn.

## Pretrained scGPT Model Zoo

Here is the list of pretrained models. Please find the links for downloading the checkpoint folders. We recommend using the `whole-human` model for most applications by default. If your fine-tuning dataset shares similar cell type context with the training data of the organ-specific models, these models can usually demonstrate competitive performance as well. A paired vocabulary file mapping gene names to ids is provided in each checkpoint folder. If ENSEMBL ids are needed, please find the conversion at [gene_info.csv](https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv).

| Model name                | Description                                             | Download                                                                                     |
| :------------------------ | :------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
| whole-human (recommended) | Pretrained on 33 million normal human cells.            | [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
| continual pretrained      | For zero-shot cell embedding related tasks.             | [link](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing) |
| brain                     | Pretrained on 13.2 million brain cells.                 | [link](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |
| blood                     | Pretrained on 10.3 million blood and bone marrow cells. | [link](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
| heart                     | Pretrained on 1.8 million heart cells                   | [link](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |
| lung                      | Pretrained on 2.1 million lung cells                    | [link](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |
| kidney                    | Pretrained on 814 thousand kidney cells                 | [link](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |
| pan-cancer                | Pretrained on 5.7 million cells of various cancer types | [link](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |

## Fine-tune scGPT for scRNA-seq integration

Please see our example code in [examples/finetune_integration.py](examples/finetune_integration.py). By default, the script assumes the scGPT checkpoint folder stored in the `examples/save` directory.

## To-do-list

- [x] Upload the pretrained model checkpoint
- [x] Publish to pypi
- [ ] Provide the pretraining code with generative attention masking
- [ ] Finetuning examples for multi-omics integration, cell type annotation, perturbation prediction, cell generation
- [x] Example code for Gene Regulatory Network analysis
- [x] Documentation website with readthedocs
- [x] Bump up to pytorch 2.0
- [x] New pretraining on larger datasets
- [x] Reference mapping example
- [ ] Publish to huggingface model hub

## Contributing

We greatly welcome contributions to scGPT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using scGPT.

## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

## Citing scGPT

```bibtex
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```
