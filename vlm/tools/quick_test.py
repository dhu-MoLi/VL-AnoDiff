#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test script for the image noise generator."""

import numpy as np
import cv2
from image_noise_generator import ImageNoiseGenerator
import os
from pathlib import Path


def create_test_image():
    """Create a simple synthetic test image."""
    test_image = np.zeros((200, 200, 3), dtype=np.uint8)

    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(test_image, (100, 100), 30, (128, 128, 128), -1)
    cv2.putText(test_image, "TEST", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return test_image


def test_noise_generation():
    """Test basic noise generation."""
    print("Creating test image...")
    test_image = create_test_image()

    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    test_image_path = test_dir / "test_image.png"
    cv2.imwrite(str(test_image_path), test_image)
    print(f"Test image saved: {test_image_path}")

    generator = ImageNoiseGenerator(output_dir="test_output")
    generator.config.update({
        "noise_levels": [0.1, 0.2, 0.3],
        "noise_rounds": [1, 2],
        "noise_type": "gaussian",
        "naming_convention": "simple",
        "create_comparison": True
    })

    print("\nTesting noise generation...")
    generated_files = generator.process_single_image(str(test_image_path))

    print(f"\nGenerated {len(generated_files)} files:")
    for file_path in generated_files:
        filename = Path(file_path).name
        print(f"  {filename}")

    generator.create_comparison_image(str(test_image_path), generated_files)

    results = {
        "total_images": 1,
        "processed_images": 1,
        "failed_images": 0,
        "generated_files": generated_files,
        "processing_time": 0
    }
    generator.generate_report(results, "test_report.txt")

    print("\nTest completed.")
    print(f"Output directory: {test_dir.absolute()}")


def test_different_noise_types():
    """Test different noise types."""
    print("\nTesting different noise types...")

    test_image = create_test_image()
    test_dir = Path("test_noise_types")
    test_dir.mkdir(exist_ok=True)
    test_image_path = test_dir / "test_image.png"
    cv2.imwrite(str(test_image_path), test_image)

    noise_types = ["gaussian", "uniform", "salt_pepper"]

    for noise_type in noise_types:
        print(f"\nTesting {noise_type} noise:")

        generator = ImageNoiseGenerator(output_dir=f"test_noise_types/{noise_type}")
        generator.config.update({
            "noise_levels": [0.2],
            "noise_rounds": [1],
            "noise_type": noise_type,
            "naming_convention": "simple"
        })

        generated_files = generator.process_single_image(str(test_image_path))
        print(f"  Generated {len(generated_files)} files")


def main():
    """Run all quick tests."""
    print("Image Noise Generator — Quick Test")
    print("=" * 40)

    try:
        test_noise_generation()
        test_different_noise_types()

        print("\n" + "=" * 40)
        print("All tests completed.")
        print("Check results in test_output/ and test_noise_types/")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
