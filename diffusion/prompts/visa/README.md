# VisA Prompt Files

## Flat format (for training)

Place aggregated prompt files here:

```
{category}_prompts.txt
```

These are consumed by `LLMEnhancedEmbeddingManager` during embedding training.

## Per-image format (raw)

The `raw/` subdirectory contains per-image prompts generated during experiments:

```
raw/{category}/prompts/{category}_{id}_prompts.txt
```

## How to Generate

Use the VLM module in VL-AnoDiff:

→ [`../vlm/prompt_generation.py`](https://github.com/dhu-MoLi/VL-AnoDiff/tree/master/vlm/prompt_generation.py)

Then aggregate or convert to the flat `{category}_prompts.txt` format before training.
