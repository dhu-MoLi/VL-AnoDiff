<div align="center">
<h1>VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis</h1>
<br>
<br>

<p>
<a>Mo Li</a>, <a>Shubo Zhou</a>, <a>Weiyu Hu</a>, <a>Xue-Qin Jiang</a>, <a>Yongbin Gao</a>

</div>


Welcome to the official repository for **VL-AnoDiff**, a novel diffusion framework for generating anomalous image–mask pairs.

## Overview

This repository contains the implementation of VL-AnoDiff, a novel diffusion framework for generating anomalous image–mask pairs. Our approach leveraged both visual and semantic features of reference samples, while employing semantic anchors to constrain the generated anomalies to be realistic and plausible.

## Todo(Latest update:2026/3/6)

We are actively working on releasing the complete codebase. The current release status is as follows:

- [x] Repository setup
- [x] [VLM-related code](vlm/)
- [ ] Model training code
- [ ] Model inference code
- [ ] Pre-trained weights

## VLM Module

The VLM semantic guidance module is available in the [`vlm/`](vlm/) directory. It uses Qwen2.5-VL to analyze defect semantics and generate mask transformation strategies and diffusion prompts.

See [`vlm/README.md`](vlm/README.md) for installation and usage instructions.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{vlanodiff2024,
  title={VL-AnoDiff: Vision-Language Guided Diffusion for Few-Shot Industrial Anomaly Synthesis},
  author={Mo Li, Shubo Zhou, Weiyu Hu, Xue-Qin Jiang, Yongbin Gao},
}
```

## License

[To be determined]

## Contact

For questions and inquiries, please open an issue or contact the authors.

