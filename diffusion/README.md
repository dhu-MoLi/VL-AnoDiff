# VL-AnoDiff — Diffusion Module

This module implements **Semantic Anchor Regularization (SAR)** and mask-guided **inpaint anomaly synthesis**, built on the [AnoGen](https://github.com/csgaobb/AnoGen) (ECCV 2024) baseline.

| Baseline (AnoGen) | VL-AnoDiff (Ours) |
|-------------------|-------------------|
| Single-word (`"defect"`) embedding init | VLM semantic anchor mixed initialization |
| No semantic constraint | SAR: MSE regularization toward VLM anchors |
| Manual bounding-box masks | SAMS masks from [`../vlm/`](../vlm/) |

---

## Pipeline

```
VLM prompts (../vlm/)  →  SAR text inversion training  →  Inpaint synthesis
```

### 1. Prepare data

```bash
bash shell/make_trainset.sh
```

### 2. Train embeddings with SAR

```bash
bash shell/run_llm_training.sh mvtec   # MVTec AD
bash shell/run_llm_training.sh visa    # VisA
```

Key arguments in `main.py`:

| Argument | Description |
|----------|-------------|
| `--use_llm_enhancement True` | Enable SAR (semantic anchor regularization) |
| `--prompt_dir prompts/mvtec` | VLM-generated prompt directory |
| `--llm_weight 0.5` | Anchor blending weight |
| `--use_adaptive_weight True` | Adaptive weight by prompt quality |

### 3. Generate anomalies

```bash
bash shell/inference_single.sh
```

Or batch generation:

```bash
python txt2image_manager.py \
    --dataset_type visa \
    --embedding_path logs/training/ \
    --mask_prompt data/masks/ \
    --image_prompt data/visa/ \
    --outname outputs/generated
```

---

## Directory Layout

```
diffusion/
├── main.py                          # SAR training entry
├── llm_patches.py                   # Inject L_anchor into training loss
├── llm_weight_adapter.py            # Adaptive anchor weight
├── txt2image_manager.py             # Batch inpaint
├── ldm/modules/embedding/
│   └── llm_enhanced_embedding_manager.py   # SAR implementation
├── configs/latent-diffusion/
│   └── txt2img-1p4B-finetune-llm.yaml
├── shell/                           # Runnable scripts
├── prompts/                         # Pre-generated VLM prompts
└── examples/demo/                   # Quick-start demo images
```

---

## Core Implementation (SAR)

**Semantic Anchor Regularization** (Eq. 5 in paper):

```
L_anchor = ||E_i - E_o||² + (1/n) Σ ||E_i - E_o^(k)||²
L_total  = L_diff + L_anchor
```

Code mapping:

| Paper | Code |
|-------|------|
| Semantic anchors E_o | `llm_enhanced_embedding_manager.py` → `_get_llm_embeddings()` |
| L_anchor | `llm_regularization_loss()` |
| Loss injection | `llm_patches.py` → `apply_llm_patches()` |

See [`docs/llm_regularization.md`](docs/llm_regularization.md) for details.

---

## Acknowledgements

- [AnoGen](https://github.com/csgaobb/AnoGen) — Baseline text inversion + inpaint framework
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) — Pre-trained diffusion model
