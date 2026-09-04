import os
import shutil
import re
from pathlib import Path


def natural_sort_key(s):
    """Key function for natural sorting (numeric order within filenames)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def copy_top_images(source_dir, dest_dir, limit=1000):
    """
    Copy up to `limit` images from source_dir to dest_dir.

    Args:
        source_dir (str): Source image directory path.
        dest_dir (str): Destination directory path.
        limit (int): Maximum number of images to copy (default: 1000).
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(source_dir)
                   if os.path.isfile(os.path.join(source_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    sorted_images = sorted(image_files, key=natural_sort_key)

    count = 0
    for image in sorted_images[:limit]:
        try:
            src_path = os.path.join(source_dir, image)
            dst_path = os.path.join(dest_dir, image)
            shutil.copy2(src_path, dst_path)
            count += 1
            if count % 100 == 0:
                print(f"Copied {count}/{min(limit, len(sorted_images))} images")
        except Exception as e:
            print(f"Failed to copy {image}: {e}")

    print(f"Done. Copied {count} images to {dest_dir}")


if __name__ == "__main__":
    type_names = [
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]

    for typename in type_names:
        print(f"\nProcessing type: {typename}")
        for name in ["ok"]:
            print(f"  Processing category: {name}")
            if name == 'ok':
                source_directory = f'datasets/visa_train_data_10/{typename}/{name}'
                destination_directory = f'generated_no_extra_prompt/{typename}/ok'
            else:
                source_directory = f'visa_train_data_10/{typename}/{name}'
                destination_directory = f'PRN_visa_10/{typename}/ko_mask'

            if not os.path.exists(source_directory):
                print(f"  Warning: source directory {source_directory} does not exist, skipping")
                continue

            try:
                copy_top_images(source_directory, destination_directory)
            except Exception as e:
                print(f"  Error processing {typename}/{name}: {e}")
