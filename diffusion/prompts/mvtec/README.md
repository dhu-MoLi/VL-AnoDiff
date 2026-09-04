# MVTec Prompt Files

Each file follows the naming convention:

```
{category}_{defect_type}_prompts.txt
```

Example: `bottle_broken_large_prompts.txt`

## Format

```
Original prompt: <physical description>
Generated prompt 1: <visual description>
Generated prompt 2: <contextual description>
```

## How to Generate

Prompts are produced by the **VLM module** in the VL-AnoDiff repository:

→ [`../vlm/prompt_generation.py`](https://github.com/dhu-MoLi/VL-AnoDiff/tree/master/vlm/prompt_generation.py)

The files in this directory are pre-generated examples for MVTec AD (73 defect types across 15 categories).
