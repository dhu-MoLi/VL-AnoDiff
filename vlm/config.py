"""Shared configuration for VLM-guided anomaly synthesis."""

import os

# Default model path: local directory or HuggingFace model ID.
# Override via environment variable: export VLM_MODEL_PATH=/path/to/model
DEFAULT_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    "./models/Qwen/Qwen2.5-VL-7B-Instruct",
)

# HuggingFace model ID (alternative to local path):
# DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
