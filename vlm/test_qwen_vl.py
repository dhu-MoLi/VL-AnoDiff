"""Quick test script for Qwen2.5-VL inference."""

import argparse

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from config import DEFAULT_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL local inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="VLM model path or HuggingFace model ID (default: see config.py)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image URL or local path for the test prompt",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Text prompt sent to the VLM",
    )
    args = parser.parse_args()

    model_path = args.model_path or DEFAULT_MODEL_PATH
    print(f"Loading model: {model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
