# Image Noise Generator

A utility for adding configurable noise to images, designed for academic paper figure preparation.

## Features

- **Multiple noise types**: Gaussian, uniform, and salt-and-pepper
- **Configurable parameters**: noise intensity, rounds, standard deviation
- **Batch processing**: single image or entire directory
- **Paper-style naming**: automatic renaming following figure conventions
- **Comparison figures**: side-by-side original vs. noisy images
- **Processing reports**: detailed logs and statistics

## Installation

```bash
pip install opencv-python numpy pillow matplotlib seaborn
```

## Usage

### Basic

```bash
# Single image
python image_noise_generator.py input_image.jpg

# Entire directory
python image_noise_generator.py input_directory/

# Custom output directory
python image_noise_generator.py input_image.jpg -o output_directory
```

### Advanced

```bash
# Use a config file
python image_noise_generator.py input.jpg -c noise_config.json

# Custom noise parameters
python image_noise_generator.py input.jpg -l 0.1 0.3 0.5 -r 1 2 3

# Specify noise type
python image_noise_generator.py input.jpg -t uniform

# Skip comparison figure
python image_noise_generator.py input.jpg --no-comparison

# Do not save original image
python image_noise_generator.py input.jpg --no-original
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input image path or directory | Required |
| `-o, --output` | Output directory | `noise_images` |
| `-c, --config` | Config file path | None |
| `-l, --levels` | Noise intensity levels | `[0.1, 0.2, 0.3, 0.4, 0.5]` |
| `-r, --rounds` | Noise application rounds | `[1, 2, 3, 4, 5]` |
| `-t, --type` | Noise type | `gaussian` |
| `--no-comparison` | Skip comparison figure | False |
| `--no-original` | Do not save original | False |

## Configuration File

Create `noise_config.json` to customize parameters:

```json
{
    "noise_levels": [0.1, 0.2, 0.3],
    "noise_rounds": [1, 2, 3],
    "noise_type": "gaussian",
    "gaussian_std": 0.1,
    "preserve_original": true,
    "naming_convention": "paper_style",
    "create_comparison": true,
    "comparison_grid_size": [2, 3]
}
```

### Parameter Reference

- **noise_levels**: intensity values, typically 0.0–1.0
- **noise_rounds**: number of noise application passes (higher = stronger effect)
- **noise_type**: `gaussian`, `uniform`, or `salt_pepper`
- **gaussian_std**: standard deviation for Gaussian noise
- **preserve_original**: whether to save the original image
- **naming_convention**: `paper_style`, `simple`, or `detailed`
- **create_comparison**: whether to generate a comparison figure
- **comparison_grid_size**: grid layout for the comparison figure

## Output Naming Conventions

### paper_style
- Original: `Figure_image_original.png`
- Noisy: `Figure_image_noise_0.2_r2.png`

### simple
- Original: `image_original.png`
- Noisy: `image_noise_0.2_2.png`

### detailed
- Original: `image_original.png`
- Noisy: `image_gaussian_noise_0.2_rounds_2.png`

## Output Structure

```
noise_images/
├── Figure_image_original.png
├── Figure_image_noise_0.1_r1.png
├── Figure_image_noise_0.2_r2.png
├── comparison_image.png
└── processing_report.txt
```

## License

MIT License — part of the VL-AnoDiff project.
