# VL-AnoDiff — VLM Module

> VLM semantic guidance and SAMS module of **[VL-AnoDiff](https://github.com/dhu-MoLi/VL-AnoDiff)** — uses a vision-language model to understand defect semantics, generate mask transformation strategies, and produce diffusion prompts.

**VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis**

Mo Li, Shubo Zhou, Weiyu Hu, Xue-Qin Jiang, Yongbin Gao

A VLM-based tool for industrial anomaly data augmentation. It analyzes defect images with **Qwen2.5-VL**, recommends mask generation strategies, and produces diffusion prompts for synthetic anomaly sample generation.

## Relationship to the Main Project

| Module | Location | Status |
|--------|----------|--------|
| VLM semantic guidance + SAMS | `vlm/` (this directory) | ✅ Complete |
| Diffusion training + SAR | [`../diffusion/`](../diffusion/) | ✅ Complete |
| Diffusion inference | [`../diffusion/`](../diffusion/) | ✅ Complete |

## Features

| Module | Script | Description |
|--------|--------|-------------|
| Mask generation | `mask_generation.py` | VLM analyzes defect types and applies recommended mask transformation strategies |
| Prompt generation | `prompt_generation.py` | VLM analyzes defect images and generates diffusion model prompts |
| Environment test | `test_qwen_vl.py` | Quick check that the VLM model loads and runs |

## Requirements

- Python >= 3.10
- CUDA GPU, >= 24GB VRAM recommended (Qwen2.5-VL-7B inference)
- Example virtual environment: `conda activate qwen`

## Installation

```bash
git clone <this-repo-url>
cd VL-AnoDiff-VLM   # or your chosen directory name

pip install -r requirements.txt
```

Main project: [https://github.com/dhu-MoLi/VL-AnoDiff](https://github.com/dhu-MoLi/VL-AnoDiff)

## Model Setup

The default model path is `./models/Qwen/Qwen2.5-VL-7B-Instruct`. Choose one of the following:

**Option 1: Download to a local directory**

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen/Qwen2.5-VL-7B-Instruct
```

**Option 2: Use the HuggingFace model ID directly**

```bash
export VLM_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
```

Or pass `--model_path` at runtime.

## Data Preparation

### MVTec AD

```
data_root/
├── bottle/
│   ├── broken_large/
│   │   └── *.png
│   └── ...
└── ...

mask_dir/
├── bottle/
│   ├── broken_large_mask/
│   │   └── *.png
│   └── ...
└── ...
```

### VisA

```
data_root/
├── candle/
│   └── ko/
│       └── *
└── ...

mask_dir/
├── candle/
│   └── ko_mask/
│       └── *
└── ...
```

## Quick Start

### 1. Test the VLM environment

```bash
python test_qwen_vl.py --model_path ./models/Qwen/Qwen2.5-VL-7B-Instruct
```

### 2. Generate masks

```bash
# MVTec example
python mask_generation.py \
  --data_root /path/to/mvtec_train_data \
  --mask_dir /path/to/mvtec_train_data \
  --output_dir output/masks_mvtec \
  --dataset mvtec \
  --num_simple 5 \
  --num_complex 5 \
  --selected_sample bottle \
  --selected_defect broken_large

# VisA example
python mask_generation.py \
  --data_root /path/to/visa_train_data \
  --mask_dir /path/to/visa_train_data \
  --output_dir output/masks_visa \
  --dataset visa \
  --num_simple 5 \
  --num_complex 5 \
  --selected_sample candle
```

### 3. Generate diffusion prompts

```bash
python prompt_generation.py \
  --data_root /path/to/visa_train_data \
  --mask_dir output/masks_visa \
  --output_dir output/prompts_visa \
  --dataset visa \
  --selected_sample candle \
  --max_samples 10
```

## Mask Generation Strategies

The VLM classifies defects as **structural** or **logical** and recommends the following strategies:

| Type | Strategies |
|------|------------|
| Structural | `elastic_deformation`, `texture_modification`, `edge_enhancement`, `fracture_simulation` |
| Logical | `translation`, `rotation`, `component_removal`, `component_addition` |

## Project Structure

```
VLM-opensource/
├── config.py                 # Shared configuration (model path, etc.)
├── mask_generation.py        # Mask generation entry point
├── prompt_generation.py      # Prompt generation entry point
├── test_qwen_vl.py           # VLM environment test
├── requirements.txt
├── LICENSE
├── README.md
├── models/                   # Model weights (download separately)
├── examples/                 # Sample outputs
│   ├── sample_config/
│   ├── sample_prompts/
│   └── sample_masks/
├── scripts/
│   └── copy_images.py        # Batch image copy utility
└── tools/                    # Auxiliary tools (paper figure noise generator, etc.)
    ├── image_noise_generator.py
    ├── example_usage.py
    └── quick_test.py
```

## Configuration

| Method | Example |
|--------|---------|
| Environment variable | `export VLM_MODEL_PATH=/path/to/model` |
| CLI argument | `--model_path Qwen/Qwen2.5-VL-7B-Instruct` |
| Default | `DEFAULT_MODEL_PATH` in `config.py` |

## Example Outputs

The `examples/` directory contains sample outputs for the VisA `candle` category:

- `sample_masks/candle/` — 3 generated masks
- `sample_prompts/candle_prompts.txt` — diffusion prompts
- `sample_config/generation_params.json` — run parameter log

## Citation

If you find this code useful, please cite VL-AnoDiff:

```bibtex
@article{vlanodiff2024,
  title   = {VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis},
  author  = {Mo Li and Shubo Zhou and Weiyu Hu and Xue-Qin Jiang and Yongbin Gao},
  year    = {2024}
}
```

## License

Copyright belongs to the VL-AnoDiff authors (Mo Li, Shubo Zhou, Weiyu Hu, Xue-Qin Jiang, Yongbin Gao).

The main project license is not yet finalized (see [VL-AnoDiff README](https://github.com/dhu-MoLi/VL-AnoDiff)). This module is temporarily released under MIT License and will be aligned with the main repository when finalized.

## Contact

For questions and suggestions, please open an issue in this repository or at [VL-AnoDiff Issues](https://github.com/dhu-MoLi/VL-AnoDiff/issues).
