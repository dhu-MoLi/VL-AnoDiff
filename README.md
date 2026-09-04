# VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis

Mo Li, Shubo Zhou, Weiyu Hu, Xue-Qin Jiang, Yongbin Gao

**IEEE ICASSP 2026** | DOI: [10.1109/ICASSP55912.2026.11461878](https://doi.org/10.1109/icassp55912.2026.11461878)

---

## Overview

**VL-AnoDiff** is a vision-language guided diffusion framework for few-shot industrial anomaly synthesis. Given only a few anomalous image–mask pairs, it generates high-fidelity, semantically controllable anomaly–mask pairs for downstream inspection tasks.

> **Note:** This project is built upon [AnoGen](https://github.com/csgaobb/AnoGen) (ECCV 2024) as the **baseline** diffusion backbone. Our contributions — **Semantic Anchor Regularization (SAR)** and **Semantically Aligned Mask Synthesis (SAMS)** — extend the baseline with vision-language guidance.

<p align="center">
  <img src="examples/fig1_comparison.png" width="860" alt="Comparison with AnoDiff and AnoGen baselines"/>
</p>

<p align="center">
  <img src="examples/fig2_framework.png" width="860" alt="VL-AnoDiff framework"/>
</p>

---

## Key Contributions

| Module | Paper Name | Code Location | Description |
|--------|-----------|---------------|-------------|
| **SAR** | Semantic Anchor Regularization | [`diffusion/`](diffusion/) | Aligns learnable embeddings with VLM-derived semantic anchors during text inversion |
| **SAMS** | Semantically Aligned Mask Synthesis | [`vlm/mask_generation.py`](vlm/mask_generation.py) | Generates diverse, semantically meaningful masks without bounding-box supervision |
| **VLM Prompts** | Auxiliary Semantic Anchor Generation | [`vlm/prompt_generation.py`](vlm/prompt_generation.py) | Qwen2.5-VL analyzes defects and produces diffusion prompts |

---

## Repository Structure

```
VL-AnoDiff/
├── vlm/                  # SAMS + VLM prompt generation (Qwen2.5-VL)
├── diffusion/            # SAR + text inversion training + inpaint inference
├── docs/
│   └── paper.pdf         # Full paper
├── examples/             # Paper figures and demo assets
└── README.md
```

---

## Release Status

| Component | Status |
|-----------|--------|
| VLM module (`vlm/`) | ✅ Released |
| Diffusion training (`diffusion/`) | ✅ Released |
| Diffusion inference (`diffusion/`) | ✅ Released |
| Pre-trained weights | 🔜 Coming soon |

---

## Quick Start

### 1. Environment

```bash
# VLM module (mask + prompt generation)
cd vlm && pip install -r requirements.txt

# Diffusion module (SAR training + inpaint)
cd diffusion && pip install -r requirements.txt
```

### 2. Download pre-trained LDM

```bash
cd diffusion
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

### 3. Full pipeline

```bash
# Step 1: VLM — generate prompts and masks
cd vlm
python prompt_generation.py
python mask_generation.py

# Step 2: Diffusion — SAR-enhanced embedding training
cd ../diffusion
bash shell/make_trainset.sh
bash shell/run_llm_training.sh visa    # or mvtec

# Step 3: Inpaint anomaly synthesis
bash shell/inference_single.sh
```

See module READMEs for detailed instructions:
- [`vlm/README.md`](vlm/README.md)
- [`diffusion/README.md`](diffusion/README.md)

---

## Results

<p align="center">
  <img src="examples/fig3_visa_results.png" width="860" alt="VisA generation results (Fig. 3)"/>
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026vlanodiff,
  title={VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis},
  author={Li, Mo and Zhou, Shubo and Hu, Weiyu and Jiang, Xue-Qin and Gao, Yongbin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

---

## Acknowledgements

This work builds upon the following open-source projects:

- [AnoGen](https://github.com/csgaobb/AnoGen) — Few-Shot Anomaly-Driven Generation (ECCV 2024), used as our diffusion baseline
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion) — Pre-trained text-to-image model
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Vision-language model for semantic understanding

---

## License

[To be determined]

## Contact

For questions, please open an issue or contact the authors.
