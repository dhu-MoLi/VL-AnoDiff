#!/bin/bash
# Single-image inpaint inference example
# Run from the diffusion/ project root

cd "$(dirname "$0")/.."

python scripts/inference/txt2img.py \
    --ddim_eta 0.0 \
    --n_samples 1 \
    --n_iter 2 \
    --scale 10.0 \
    --ddim_steps 50 \
    --embedding_path "logs/training/bottle_broken_large/checkpoints/embeddings.pt" \
    --ckpt_path "models/ldm/text2img-large/model.ckpt" \
    --prompt "*" \
    --mask_prompt "examples/demo/mask.png" \
    --image_prompt "examples/demo/normal.png" \
    --outdir "outputs/demo"
