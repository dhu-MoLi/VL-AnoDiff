#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Noise Generator for academic paper illustrations.

Features:
1. Load specified images
2. Add Gaussian / uniform / salt-and-pepper noise
3. Save with paper-style naming conventions
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ImageNoiseGenerator:
    """Add configurable noise to images for paper figure generation."""

    def __init__(self, output_dir: str = "noise_images", config_file: Optional[str] = None):
        """Initialize the noise generator.

        Args:
            output_dir: Output directory path.
            config_file: Optional path to a JSON config file.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.config = {
            "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "noise_rounds": [1, 2, 3, 4, 5],
            "noise_type": "gaussian",
            "gaussian_std": 0.1,
            "preserve_original": True,
            "image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
            "output_format": ".png",
            "naming_convention": "paper_style",
            "create_comparison": True,
            "comparison_grid_size": (2, 3),
        }

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

    def load_config(self, config_file: str):
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self.config.update(user_config)
            print(f"Loaded config: {config_file}")
        except Exception as e:
            print(f"Failed to load config: {e}")

    def save_config(self, config_file: str):
        """Save current configuration to a JSON file."""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"Config saved to: {config_file}")
        except Exception as e:
            print(f"Failed to save config: {e}")

    def add_gaussian_noise(self, image: np.ndarray, std: float, rounds: int = 1) -> np.ndarray:
        """Add Gaussian noise to an image."""
        noisy_image = image.copy().astype(np.float32)

        for _ in range(rounds):
            noise = np.random.normal(0, std * 255, image.shape).astype(np.float32)
            noisy_image += noise

        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    def add_uniform_noise(self, image: np.ndarray, intensity: float, rounds: int = 1) -> np.ndarray:
        """Add uniform noise to an image."""
        noisy_image = image.copy().astype(np.float32)

        for _ in range(rounds):
            noise = np.random.uniform(-intensity * 255, intensity * 255, image.shape).astype(np.float32)
            noisy_image += noise

        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray, intensity: float, rounds: int = 1) -> np.ndarray:
        """Add salt-and-pepper noise to an image."""
        noisy_image = image.copy()

        for _ in range(rounds):
            random_mask = np.random.random(image.shape[:2])

            salt_mask = random_mask < intensity / 2
            noisy_image[salt_mask] = 0

            pepper_mask = random_mask > 1 - intensity / 2
            noisy_image[pepper_mask] = 255

        return noisy_image

    def generate_filename(self, original_name: str, noise_level: float, rounds: int,
                         noise_type: str, naming_convention: str = "paper_style") -> str:
        """Generate an output filename based on the naming convention."""
        base_name = Path(original_name).stem

        if naming_convention == "paper_style":
            if rounds == 0:
                return f"Figure_{base_name}_original{self.config['output_format']}"
            return f"Figure_{base_name}_noise_{noise_level:.1f}_r{rounds}{self.config['output_format']}"

        elif naming_convention == "simple":
            if rounds == 0:
                return f"{base_name}_original{self.config['output_format']}"
            return f"{base_name}_noise_{noise_level:.1f}_{rounds}{self.config['output_format']}"

        elif naming_convention == "detailed":
            if rounds == 0:
                return f"{base_name}_original{self.config['output_format']}"
            return f"{base_name}_{noise_type}_noise_{noise_level:.1f}_rounds_{rounds}{self.config['output_format']}"

        return f"{base_name}_noise_{noise_level:.1f}_{rounds}{self.config['output_format']}"

    def process_single_image(self, image_path: str, noise_levels: List[float] = None,
                           noise_rounds: List[int] = None) -> List[str]:
        """Process a single image and return paths of generated files."""
        if noise_levels is None:
            noise_levels = self.config["noise_levels"]
        if noise_rounds is None:
            noise_rounds = self.config["noise_rounds"]

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Cannot read image: {image_path}")
                return []

            print(f"Processing image: {image_path}")
            print(f"  Image shape: {image.shape}")

        except Exception as e:
            print(f"Failed to read image: {e}")
            return []

        generated_files = []

        if self.config["preserve_original"]:
            original_filename = self.generate_filename(
                image_path, 0, 0, "original", self.config["naming_convention"]
            )
            original_path = self.output_dir / original_filename
            cv2.imwrite(str(original_path), image)
            generated_files.append(str(original_path))
            print(f"  Saved original: {original_filename}")

        for noise_level in noise_levels:
            for rounds in noise_rounds:
                if self.config["noise_type"] == "gaussian":
                    noisy_image = self.add_gaussian_noise(image, noise_level, rounds)
                elif self.config["noise_type"] == "uniform":
                    noisy_image = self.add_uniform_noise(image, noise_level, rounds)
                elif self.config["noise_type"] == "salt_pepper":
                    noisy_image = self.add_salt_pepper_noise(image, noise_level, rounds)
                else:
                    print(f"Unknown noise type: {self.config['noise_type']}")
                    continue

                filename = self.generate_filename(
                    image_path, noise_level, rounds, self.config["noise_type"],
                    self.config["naming_convention"]
                )
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), noisy_image)
                generated_files.append(str(output_path))

                print(f"  Saved noisy image: {filename} (level:{noise_level:.1f}, rounds:{rounds})")

        return generated_files

    def create_comparison_image(self, image_path: str, generated_files: List[str]):
        """Create a side-by-side comparison figure."""
        if not self.config["create_comparison"]:
            return

        try:
            original_image = cv2.imread(image_path)
            if original_image is None:
                return

            display_images = [original_image]
            display_labels = ["Original"]

            for file_path in generated_files[:5]:
                if "original" not in file_path:
                    img = cv2.imread(file_path)
                    if img is not None:
                        display_images.append(img)
                        filename = Path(file_path).stem
                        display_labels.append(filename.split('_')[-2:])

            rows, cols = self.config["comparison_grid_size"]
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            fig.suptitle(f'Image Noise Comparison - {Path(image_path).name}', fontsize=16)

            for i, (img, label) in enumerate(zip(display_images, display_labels)):
                if i >= rows * cols:
                    break

                row, col = i // cols, i % cols
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes
                else:
                    ax = axes[row, col] if cols > 1 else axes[row]

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(' '.join(label) if isinstance(label, list) else label)
                ax.axis('off')

            for i in range(len(display_images), rows * cols):
                row, col = i // cols, i % cols
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes
                else:
                    ax = axes[row, col] if cols > 1 else axes[row]
                ax.axis('off')

            plt.tight_layout()

            comparison_filename = f"comparison_{Path(image_path).stem}.png"
            comparison_path = self.output_dir / comparison_filename
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved comparison figure: {comparison_filename}")

        except Exception as e:
            print(f"Failed to create comparison figure: {e}")

    def process_batch(self, image_paths: List[str], noise_levels: List[float] = None,
                     noise_rounds: List[int] = None) -> dict:
        """Process a batch of images."""
        results = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "generated_files": [],
            "processing_time": 0
        }

        start_time = datetime.now()

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")

            try:
                generated_files = self.process_single_image(image_path, noise_levels, noise_rounds)
                results["generated_files"].extend(generated_files)
                results["processed_images"] += 1
                self.create_comparison_image(image_path, generated_files)

            except Exception as e:
                print(f"Failed to process image: {e}")
                results["failed_images"] += 1

        end_time = datetime.now()
        results["processing_time"] = (end_time - start_time).total_seconds()

        return results

    def generate_report(self, results: dict, report_file: str = None):
        """Write a processing report to disk."""
        if report_file is None:
            report_file = self.output_dir / "processing_report.txt"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("Image Noise Generation Processing Report\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total images: {results['total_images']}\n")
                f.write(f"Processed: {results['processed_images']}\n")
                f.write(f"Failed: {results['failed_images']}\n")
                f.write(f"Elapsed time: {results['processing_time']:.2f} s\n\n")

                f.write("Configuration:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.config.items():
                    f.write(f"{key}: {value}\n")

                f.write(f"\nGenerated files ({len(results['generated_files'])}):\n")
                f.write("-" * 30 + "\n")
                for file_path in results['generated_files']:
                    f.write(f"{file_path}\n")

            print(f"Report saved: {report_file}")

        except Exception as e:
            print(f"Failed to write report: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Image noise generator for paper illustrations")
    parser.add_argument("input", help="Input image path or directory")
    parser.add_argument("-o", "--output", default="noise_images", help="Output directory")
    parser.add_argument("-c", "--config", help="Path to JSON config file")
    parser.add_argument("-l", "--levels", nargs="+", type=float, help="Noise intensity levels")
    parser.add_argument("-r", "--rounds", nargs="+", type=int, help="Noise application rounds")
    parser.add_argument("-t", "--type", choices=["gaussian", "uniform", "salt_pepper"],
                       help="Noise type")
    parser.add_argument("--no-comparison", action="store_true", help="Skip comparison figure")
    parser.add_argument("--no-original", action="store_true", help="Do not save original image")

    args = parser.parse_args()

    generator = ImageNoiseGenerator(args.output, args.config)

    if args.levels:
        generator.config["noise_levels"] = args.levels
    if args.rounds:
        generator.config["noise_rounds"] = args.rounds
    if args.type:
        generator.config["noise_type"] = args.type
    if args.no_comparison:
        generator.config["create_comparison"] = False
    if args.no_original:
        generator.config["preserve_original"] = False

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_paths = []
        for ext in generator.config["image_formats"]:
            image_paths.extend(input_path.glob(f"*{ext}"))
            image_paths.extend(input_path.glob(f"*{ext.upper()}"))
        image_paths = [str(p) for p in image_paths]
    else:
        print(f"Input path does not exist: {args.input}")
        return

    if not image_paths:
        print("No valid image files found")
        return

    print(f"Found {len(image_paths)} image(s)")
    print(f"Output directory: {args.output}")
    print(f"Noise type: {generator.config['noise_type']}")
    print(f"Noise levels: {generator.config['noise_levels']}")
    print(f"Noise rounds: {generator.config['noise_rounds']}")

    results = generator.process_batch(image_paths)
    generator.generate_report(results)

    print("\nProcessing complete.")
    print(f"Processed: {results['processed_images']}/{results['total_images']} images")
    print(f"Generated files: {len(results['generated_files'])}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
