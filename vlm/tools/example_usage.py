#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usage examples for the ImageNoiseGenerator class."""

from image_noise_generator import ImageNoiseGenerator
import os
from pathlib import Path


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)

    generator = ImageNoiseGenerator(output_dir="example_output")
    generator.config.update({
        "noise_levels": [0.1, 0.2, 0.3],
        "noise_rounds": [1, 2],
        "noise_type": "gaussian",
        "naming_convention": "paper_style"
    })

    test_image = "datasets/visa/candle/000.png"
    if os.path.exists(test_image):
        print(f"Processing test image: {test_image}")
        generated_files = generator.process_single_image(test_image)
        print(f"Generated {len(generated_files)} files")
        generator.create_comparison_image(test_image, generated_files)
    else:
        print(f"Test image not found: {test_image}")


def example_custom_config():
    """Custom configuration example."""
    print("\n" + "=" * 60)
    print("Custom Configuration Example")
    print("=" * 60)

    custom_config = {
        "noise_levels": [0.05, 0.15, 0.25, 0.35],
        "noise_rounds": [1, 3, 5],
        "noise_type": "gaussian",
        "gaussian_std": 0.15,
        "preserve_original": True,
        "naming_convention": "detailed",
        "create_comparison": True,
        "comparison_grid_size": [2, 4]
    }

    generator = ImageNoiseGenerator(output_dir="custom_output")
    generator.config.update(custom_config)

    print("Custom configuration:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")

    generator.save_config("custom_noise_config.json")


def example_different_noise_types():
    """Different noise type examples."""
    print("\n" + "=" * 60)
    print("Different Noise Types Example")
    print("=" * 60)

    noise_types = ["gaussian", "uniform", "salt_pepper"]

    for noise_type in noise_types:
        print(f"\nProcessing {noise_type} noise:")

        generator = ImageNoiseGenerator(output_dir=f"noise_{noise_type}_output")
        generator.config.update({
            "noise_levels": [0.2, 0.4],
            "noise_rounds": [1, 2],
            "noise_type": noise_type,
            "naming_convention": "simple"
        })

        test_image = "datasets/visa/candle/000.png"
        if os.path.exists(test_image):
            generated_files = generator.process_single_image(test_image)
            print(f"  Generated {len(generated_files)} files")
        else:
            print(f"  Test image not found: {test_image}")


def example_batch_processing():
    """Batch processing example."""
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    test_images = []
    visa_dir = Path("datasets/visa")
    if visa_dir.exists():
        for category_dir in visa_dir.iterdir():
            if category_dir.is_dir():
                for img_file in category_dir.glob("*.png"):
                    test_images.append(str(img_file))
                    if len(test_images) >= 3:
                        break
            if len(test_images) >= 3:
                break

    if test_images:
        print(f"Found {len(test_images)} test images")

        generator = ImageNoiseGenerator(output_dir="batch_output")
        generator.config.update({
            "noise_levels": [0.1, 0.3],
            "noise_rounds": [1, 2],
            "noise_type": "gaussian",
            "create_comparison": True
        })

        results = generator.process_batch(test_images)
        generator.generate_report(results)

        print("Batch processing complete:")
        print(f"  Total images: {results['total_images']}")
        print(f"  Processed: {results['processed_images']}")
        print(f"  Failed: {results['failed_images']}")
        print(f"  Generated files: {len(results['generated_files'])}")
        print(f"  Processing time: {results['processing_time']:.2f} s")
    else:
        print("No test images found")


def example_paper_illustration():
    """Paper figure illustration example."""
    print("\n" + "=" * 60)
    print("Paper Illustration Example")
    print("=" * 60)

    paper_config = {
        "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
        "noise_rounds": [1, 2, 3],
        "noise_type": "gaussian",
        "gaussian_std": 0.1,
        "preserve_original": True,
        "naming_convention": "paper_style",
        "create_comparison": True,
        "comparison_grid_size": [2, 3],
        "output_format": ".png"
    }

    generator = ImageNoiseGenerator(output_dir="paper_figures")
    generator.config.update(paper_config)

    print("Paper illustration configuration:")
    for key, value in paper_config.items():
        print(f"  {key}: {value}")

    test_image = "datasets/visa/candle/000.png"
    if os.path.exists(test_image):
        print(f"\nProcessing paper figure: {test_image}")
        generated_files = generator.process_single_image(test_image)

        print("\nGenerated paper figure files:")
        for file_path in generated_files:
            filename = Path(file_path).name
            print(f"  {filename}")

        generator.create_comparison_image(test_image, generated_files)

        results = {
            "total_images": 1,
            "processed_images": 1,
            "failed_images": 0,
            "generated_files": generated_files,
            "processing_time": 0
        }
        generator.generate_report(results, "paper_illustration_report.txt")

    else:
        print(f"Test image not found: {test_image}")


def main():
    """Run all examples."""
    print("Image Noise Generator — Usage Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_custom_config()
        example_different_noise_types()
        example_batch_processing()
        example_paper_illustration()

        print("\n" + "=" * 60)
        print("All examples completed.")
        print("=" * 60)

    except Exception as e:
        print(f"Example run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
