# Semantic Anchor Regularization (SAR)

> VL-AnoDiff diffusion module — built on the [AnoGen](https://github.com/csgaobb/AnoGen) baseline.

## Limitations of the AnoGen Baseline

1. Initializes the trainable embedding from a single word (e.g., `"defect"`)
2. Provides limited semantic information about the defect
3. Training may drift away from a semantically valid embedding space

## VL-AnoDiff Improvements

### 1. Semantic Anchor Initialization

Instead of a single word, we blend VLM-generated defect descriptions into the initial embedding:

```
token_params = (1 - llm_weight) * init_embed + llm_weight * llm_embed
```

### 2. Semantic Anchor Regularization

During training, an MSE loss keeps the optimized embedding close to VLM anchors:

```
loss_llm_reg = MSE(optimized_embed, llm_embed)
total_loss = diffusion_loss + loss_llm_reg
```

This corresponds to **L_anchor** in Eq. (5) of the paper.

### 3. Adaptive Weighting

`llm_weight_adapter.py` adjusts `llm_weight` based on prompt quality scores, balancing semantic guidance and image fidelity.

## Code Mapping

| Paper | File |
|-------|------|
| Semantic anchors E_o | `llm_enhanced_embedding_manager.py` → `_get_llm_embeddings()` |
| L_anchor | `llm_regularization_loss()` |
| Loss injection | `llm_patches.py` → `apply_llm_patches()` |
| Adaptive weight | `llm_weight_adapter.py` |

## Relationship to the VLM Module

- **VLM module** ([`../../vlm/`](../../vlm/)): analyzes defects, generates prompts and SAMS masks
- **Diffusion module** (this directory): reads prompts, applies SAR during text inversion, runs inpaint synthesis

Together they form the full VL-AnoDiff pipeline.
