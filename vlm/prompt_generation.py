import os
import json
import time
import numpy as np
import cv2
import base64
import random
import glob
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from config import DEFAULT_MODEL_PATH


class DiffusionPromptGenerator:
    """Diffusion model prompt generator powered by a local vision-language model
    
    Analyzes defect images and masks to generate detailed prompts for diffusion models
    """
    
    def __init__(self, api_key=None, cache_dir="prompt_cache", max_image_size=512, model_path=None):
        """Initialize the generator
        
        Args:
            api_key: Deprecated, kept for backward compatibility
            cache_dir: Cache directory for storing responses
            max_image_size: Maximum image dimension (max of width or height), default 512 to reduce VRAM usage
            model_path: VLM model path or HuggingFace model ID
        """
        self.cache_dir = cache_dir
        self.cache = {}
        self.max_image_size = max_image_size  # Set default image size limit
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cache
        cache_file = os.path.join(cache_dir, "prompt_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached entries")
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}")
        
        # Load local model
        print(f"Loading Qwen2.5-VL model: {self.model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
    
    def _print_memory_usage(self, stage=""):
        """Print VRAM usage (for debugging)"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - reserved
            print(f"[VRAM Monitor {stage}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB, Total: {total:.2f}GB")
    
    def _resize_image_if_needed(self, image_path, max_size=512):
        """Resize image if needed and return temporary file path
        
        Args:
            image_path: Original image path
            max_size: Maximum dimension (max of width or height)
            
        Returns:
            Temporary file path if resize was needed; otherwise the original path
        """
        try:
            # Read image dimensions
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            height, width = img.shape[:2]
            max_dim = max(height, width)
            
            # Return original path if image fits within max_size
            if max_dim <= max_size:
                return image_path
            
            # Resize needed
            scale = max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Save to temporary file
            temp_path = os.path.join(self.cache_dir, f"temp_resized_{int(time.time() * 1000)}_{os.path.basename(image_path)}")
            cv2.imwrite(temp_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            return temp_path
            
        except Exception as e:
            print(f"Warning: Failed to resize image {image_path}: {e}")
            return image_path
    
    def _encode_image(self, image_path):
        """Encode image as a base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _create_mask_overlay(self, image_path, mask_path, output_path=None, max_size=1024):
        """Create an overlay of image and mask for analysis
        
        Args:
            image_path: Image path
            mask_path: Mask path
            output_path: Output path
            max_size: Maximum dimension (max of width or height) to limit VRAM usage
        """
        # Read image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return None
        
        # Limit image size to reduce VRAM usage
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = new_height, new_width
            
        # Ensure dimensions match
        mask = cv2.resize(mask, (width, height))
        
        # Create colored mask overlay
        overlay = image.copy()
        # Add semi-transparent red overlay on anomalous regions
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 2] = 255  # Red channel
        
        # Create three-channel mask version
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3channel.astype(np.float32) / 255.0
        
        # Apply overlay
        overlay = overlay.astype(np.float32)
        overlay = overlay * (1 - mask_normalized * 0.4) + red_overlay * mask_normalized * 0.4
        overlay = overlay.astype(np.uint8)
        
        # Save overlay image
        if output_path:
            cv2.imwrite(output_path, overlay)
        
        return overlay
    
    def _build_description_prompt(self, sample_name, defect_name, dataset_type):
        """Build prompt for generating raw defect descriptions"""
        prompt = f"""
Look at the defect shown in the white mask areas. Describe it with just a few simple words.

**Task:**
Use only 2-4 words to describe the defect. Format: [adjective(s)] + [noun]

**Examples:**
- "dark crack"
- "small hole" 
- "white spot"
- "bent wire"
- "missing part"
- "rough surface"
- "broken edge"

**Output:**
Just write the simple description (2-4 words only). No sentences, no explanations.
"""
        return prompt
    
    def _build_analysis_prompt(self, sample_name, defect_name, dataset_type):
        """Build analysis prompt"""
        prompt = f"""
As a professional industrial defect analysis expert and AI prompt engineering specialist, please analyze the provided images to generate high-quality diffusion model prompts.

**Task Context:**
- Dataset: {dataset_type.upper()}
- Product Category: {sample_name}
- Defect Type: {defect_name if defect_name != 'anomaly' else 'various anomalies'}

**Images Provided:**
1. Original defective product image
2. Defect mask (highlighting the anomalous regions)
3. Overlay image (original + mask visualization)

**Analysis Requirements:**
Please provide a comprehensive analysis including:

1. **Defect Description**: Detailed description of the visual characteristics of the defect/anomaly
2. **Appearance Details**: Color, texture, shape, size, pattern, and other visual features
3. **Context Analysis**: How the defect relates to the surrounding normal product areas
4. **Defect Category**: Structural defects (cracks, holes, contamination) vs Logical defects (missing parts, misplacement)

**Prompt Generation Requirements:**
Generate 5-8 high-quality diffusion model prompts that could recreate similar defects, including:

- **Positive prompts**: Detailed descriptions for generating the defect
- **Negative prompts**: What to avoid in generation
- **Style modifiers**: Technical photography terms, lighting, quality descriptors
- **Defect-specific terms**: Precise terminology for the type of anomaly

**Output Format:**
Please return your analysis in the following JSON format:

```json
{{
    "defect_analysis": {{
        "visual_description": "Detailed description of what you see in the defect",
        "appearance_details": {{
            "color": "Color characteristics",
            "texture": "Surface texture description", 
            "shape": "Geometric shape and boundaries",
            "size": "Relative size description",
            "pattern": "Pattern or distribution description"
        }},
        "context_analysis": "How defect relates to surrounding areas",
        "defect_category": "structural or logical",
        "severity": "low/medium/high"
    }},
    "diffusion_prompts": {{
        "positive_prompts": [
            "Prompt 1 for generating similar defect",
            "Prompt 2 with different perspective/style",
            "Prompt 3 with technical photography terms",
            "..."
        ],
        "negative_prompts": [
            "What to avoid in generation 1",
            "What to avoid in generation 2", 
            "..."
        ],
        "style_modifiers": [
            "Industrial photography, macro lens",
            "High resolution, detailed texture",
            "..."
        ],
        "technical_terms": [
            "Specific defect terminology 1",
            "Manufacturing defect term 2",
            "..."
        ]
    }}
}}
```

Please ensure all prompts are:
- Technically accurate for industrial defect generation
- Specific enough to recreate similar visual characteristics  
- Compatible with stable diffusion and similar models
- Include relevant technical and industrial terminology
"""
        return prompt
    
    def _call_llm_api(self, prompt, image_paths):
        """Run inference with the local model"""
        temp_image_paths = []  # Track temp files for cleanup
        try:
            # Clear GPU cache before processing
            torch.cuda.empty_cache()
            if hasattr(self, 'monitor_memory') and self.monitor_memory:
                self._print_memory_usage("before inference")
            
            # Preprocess images: limit size to reduce VRAM usage
            max_size = getattr(self, 'max_image_size', 512)  # Default 512, conservative
            processed_image_paths = []
            for img_path in image_paths:
                resized_path = self._resize_image_if_needed(img_path, max_size=max_size)
                processed_image_paths.append(resized_path)
                # Track temp files for cleanup after resize
                if resized_path != img_path:
                    temp_image_paths.append(resized_path)
            
            # Prepare image input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{"type": "image", "image": path} for path in processed_image_paths]
                    ]
                }
            ]
            
            # Prepare inference inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")
            
            if hasattr(self, 'monitor_memory') and self.monitor_memory:
                self._print_memory_usage("after input prep")
            
            # Use torch.no_grad() to reduce VRAM usage
            with torch.no_grad():
                # Generate output
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            
            # Explicitly delete intermediate tensors to free VRAM
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
            
            if hasattr(self, 'monitor_memory') and self.monitor_memory:
                self._print_memory_usage("after inference")
            
            result = output_text[0]
            
            # Clean up temporary image files
            for temp_path in temp_image_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    pass  # Ignore cleanup errors
            
            return result
            
        except Exception as e:
            print(f"Model inference failed: {e}")
            # Still clean VRAM and temp files on error
            torch.cuda.empty_cache()
            for temp_path in temp_image_paths:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
            if hasattr(self, 'monitor_memory') and self.monitor_memory:
                self._print_memory_usage("after error")
            return None
    
    def describe_defect_region(self, image_path, mask_path, sample_name, defect_name, dataset_type):
        """Generate a raw description of the defect region"""
        print(f"Generating raw description: {os.path.basename(image_path)}")
        
        # Create temporary overlay image
        temp_overlay_path = os.path.join(self.cache_dir, f"temp_overlay_desc_{int(time.time())}.jpg")
        max_size = getattr(self, 'max_image_size', 1024)
        overlay_image = self._create_mask_overlay(image_path, mask_path, temp_overlay_path, max_size=max_size)
        
        if overlay_image is None:
            print(f"Warning: Failed to create overlay image, skipping {image_path}")
            return None
        
        try:
            # Build description prompt
            prompt_text = self._build_description_prompt(sample_name, defect_name, dataset_type)
            
            # Prepare image path list
            image_paths = [image_path, mask_path, temp_overlay_path]
            
            # Call local model
            response_content = self._call_llm_api(prompt_text, image_paths)
            
            if response_content:
                # Clean response and extract plain description text
                description = self._extract_description_text(response_content)
                return description
            else:
                print("Model inference failed, returning empty result")
                return None
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_overlay_path):
                os.remove(temp_overlay_path)
    
    def _extract_description_text(self, response_content):
        """Extract plain description text from model response"""
        # Remove possible formatting markers
        description = response_content.strip()
        
        # Remove possible markdown formatting
        if description.startswith('```') and description.endswith('```'):
            lines = description.split('\n')
            description = '\n'.join(lines[1:-1])
        
        # Remove extra blank lines
        lines = [line.strip() for line in description.split('\n') if line.strip()]
        description = ' '.join(lines)
        
        return description
    
    def analyze_defect_images(self, image_path, mask_path, sample_name, defect_name, dataset_type):
        """Analyze defect images and generate prompts"""
        # Check cache
        cache_key = f"{sample_name}_{defect_name}_{os.path.basename(image_path)}_{os.path.basename(mask_path)}"
        if cache_key in self.cache:
            print(f"Using cached result: {cache_key}")
            return self.cache[cache_key]
        
        print(f"Analyzing image: {os.path.basename(image_path)}")
        
        # Create temporary overlay image
        temp_overlay_path = os.path.join(self.cache_dir, f"temp_overlay_{int(time.time())}.jpg")
        max_size = getattr(self, 'max_image_size', 1024)
        overlay_image = self._create_mask_overlay(image_path, mask_path, temp_overlay_path, max_size=max_size)
        
        if overlay_image is None:
            print(f"Warning: Failed to create overlay image, skipping {image_path}")
            return None
        
        try:
            # Build prompt
            prompt_text = self._build_analysis_prompt(sample_name, defect_name, dataset_type)
            
            # Prepare image path list
            image_paths = [image_path, mask_path, temp_overlay_path]
            
            # Call local model
            response_content = self._call_llm_api(prompt_text, image_paths)
            
            if response_content:
                print(f"Model response length: {len(response_content)} characters")
                
                # Parse response
                analysis_result = self._parse_analysis_response(response_content)
                
                # Cache result
                if analysis_result:
                    self.cache[cache_key] = analysis_result
                    self._save_cache()
                
                return analysis_result
            else:
                print("Model inference failed, returning empty result")
                return None
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_overlay_path):
                os.remove(temp_overlay_path)
    
    def _parse_analysis_response(self, response_content):
        """Parse analysis response"""
        try:
            # Try to extract JSON
            import re
            
            # Find JSON code block
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response_content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    pass
            
            # If no JSON code block found, try to find raw JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if "defect_analysis" in result and "diffusion_prompts" in result:
                        return result
                except json.JSONDecodeError:
                    continue
            
            # If JSON parsing fails, try to extract prompts from text
            return self._extract_prompts_from_text(response_content)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    
    def _extract_prompts_from_text(self, text):
        """Extract prompts from plain text"""
        # Base structure
        result = {
            "defect_analysis": {
                "visual_description": "",
                "appearance_details": {},
                "context_analysis": "",
                "defect_category": "structural",
                "severity": "medium"
            },
            "diffusion_prompts": {
                "positive_prompts": [],
                "negative_prompts": [],
                "style_modifiers": [],
                "technical_terms": []
            }
        }
        
        # Simple keyword extraction
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify prompt-related lines
            if any(keyword in line.lower() for keyword in ['prompt', 'generate', 'diffusion', 'stable']):
                # Extract possible prompts
                if '"' in line:
                    # Extract quoted content
                    import re
                    quotes = re.findall(r'"([^"]*)"', line)
                    for quote in quotes:
                        if len(quote) > 10:  # Filter out content that is too short
                            result["diffusion_prompts"]["positive_prompts"].append(quote)
                elif ':' in line and len(line) > 20:
                    # Extract content after colon
                    parts = line.split(':', 1)
                    if len(parts) == 2 and len(parts[1].strip()) > 10:
                        result["diffusion_prompts"]["positive_prompts"].append(parts[1].strip())
        
        # Add defaults if no prompts were extracted
        if not result["diffusion_prompts"]["positive_prompts"]:
            result["diffusion_prompts"]["positive_prompts"] = [
                "industrial defect, manufacturing anomaly, detailed macro photography",
                "product quality control, surface defect, high resolution industrial image",
                "manufacturing error, technical photography, professional lighting"
            ]
        
        return result
    
    def _save_cache(self):
        """Save cache"""
        cache_file = os.path.join(self.cache_dir, "prompt_cache.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _get_file_patterns(self, sample_name, defect_name, data_root, mask_dir, dataset_type):
        """Return file patterns based on dataset type"""
        if dataset_type == 'mvtec':
            # MVTec dataset file structure
            mask_pattern = os.path.join(mask_dir, sample_name, f"{defect_name}_mask", "*.png")
            img_pattern = os.path.join(data_root, sample_name, defect_name, "*.png")
        elif dataset_type == 'visa':
            # VISA dataset file structure
            mask_pattern = os.path.join(mask_dir, sample_name, "ko_mask", "*")
            img_pattern = os.path.join(data_root, sample_name, "ko", "*")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        return mask_pattern, img_pattern
    
    def generate_prompts_for_dataset(self, sample_name, defect_name, data_root, mask_dir, 
                                   output_dir, dataset_type='mvtec', max_samples=None):
        """Generate prompts for the specified dataset"""
        print(f"\n==== Generating prompts for {dataset_type.upper()} dataset {sample_name}/{defect_name} ====")
        
        # Get file patterns
        mask_pattern, img_pattern = self._get_file_patterns(
            sample_name, defect_name, data_root, mask_dir, dataset_type
        )
        
        # Collect files
        mask_files = glob.glob(mask_pattern)
        img_files = glob.glob(img_pattern)
        
        print(f"Found {len(mask_files)} mask files")
        print(f"Found {len(img_files)} image files")
        
        if not mask_files or not img_files:
            print(f"Warning: Not enough files found")
            print(f"Mask pattern: {mask_pattern}")
            print(f"Image pattern: {img_pattern}")
            return []
        
        # Match image and mask files
        matched_pairs = []
        for mask_file in mask_files:
            mask_basename = os.path.splitext(os.path.basename(mask_file))[0]
            
            # Find corresponding image for each mask
            for img_file in img_files:
                img_basename = os.path.splitext(os.path.basename(img_file))[0]
                
                # Simple name matching (may need adjustment for specific datasets)
                if mask_basename == img_basename or mask_basename in img_basename or img_basename in mask_basename:
                    matched_pairs.append((img_file, mask_file))
                    break
        
        print(f"Matched {len(matched_pairs)} image-mask pairs")
        
        if not matched_pairs:
            print("Warning: No matching image-mask pairs found")
            return []
        
        # Limit sample count
        if max_samples and len(matched_pairs) > max_samples:
            import random
            matched_pairs = random.sample(matched_pairs, max_samples)
            print(f"Randomly selected {max_samples} samples for analysis")
        
        # Step 1: Generate raw descriptions
        print(f"\n==== Step 1: Generate raw defect descriptions ====")
        original_descriptions = []
        
        for i, (img_file, mask_file) in enumerate(matched_pairs):
            print(f"\nRaw description progress: {i+1}/{len(matched_pairs)}")
            
            try:
                description = self.describe_defect_region(
                    img_file, mask_file, sample_name, defect_name, dataset_type
                )
                
                if description:
                    original_descriptions.append(description)
                    print(f"Successfully generated description: {os.path.basename(img_file)}")
                else:
                    print(f"Description generation failed: {os.path.basename(img_file)}")
                
                # Clear VRAM after each sample
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                # Clear VRAM on error as well
                torch.cuda.empty_cache()
        
        # Save raw descriptions
        if original_descriptions:
            original_output_filename = f"{sample_name}_originalprompts.txt"
            original_output_path = os.path.join(output_dir, original_output_filename)
            
            with open(original_output_path, 'w', encoding='utf-8') as f:
                for desc in original_descriptions:
                    f.write(desc + '\n')
            
            print(f"\nRaw descriptions saved to: {original_output_path}")
            print(f"Generated {len(original_descriptions)} raw descriptions in total")
        
        # Step 2: Detailed analysis for diffusion model prompts
        print(f"\n==== Step 2: Generate diffusion model prompts ====")
        all_prompts = []
        successful_analyses = 0
        
        for i, (img_file, mask_file) in enumerate(matched_pairs):
            print(f"\nPrompt generation progress: {i+1}/{len(matched_pairs)}")
            
            try:
                analysis_result = self.analyze_defect_images(
                    img_file, mask_file, sample_name, defect_name, dataset_type
                )
                
                if analysis_result and "diffusion_prompts" in analysis_result:
                    # Extract all prompt types
                    prompts = analysis_result["diffusion_prompts"]
                    
                    # Collect positive prompts
                    for prompt in prompts.get("positive_prompts", []):
                        if prompt.strip():
                            all_prompts.append(prompt.strip())
                    
                    # Add technical term combinations
                    technical_terms = prompts.get("technical_terms", [])
                    style_modifiers = prompts.get("style_modifiers", [])
                    
                    # Combine technical terms and style modifiers
                    if technical_terms and style_modifiers:
                        for term in technical_terms[:2]:  # Take first 2 technical terms
                            for style in style_modifiers[:2]:  # Take first 2 style modifiers
                                combined = f"{term}, {style}"
                                all_prompts.append(combined)
                    
                    successful_analyses += 1
                    print(f"Successfully analyzed: {os.path.basename(img_file)}")
                else:
                    print(f"Analysis failed: {os.path.basename(img_file)}")
                
                # Clear VRAM after each sample
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                # Clear VRAM on error as well
                torch.cuda.empty_cache()
        
        print(f"\nSuccessfully analyzed {successful_analyses}/{len(matched_pairs)} samples")
        print(f"Generated {len(all_prompts)} prompts in total")
        
        # Deduplicate and filter
        unique_prompts = []
        seen = set()
        
        for prompt in all_prompts:
            # Simple deduplication and quality filtering
            prompt_clean = prompt.lower().strip()
            if (prompt_clean not in seen and 
                len(prompt) > 10 and 
                len(prompt) < 200 and
                prompt_clean not in ['', 'none', 'n/a']):
                unique_prompts.append(prompt)
                seen.add(prompt_clean)
        
        print(f"{len(unique_prompts)} valid prompts remaining after deduplication")
        
        # Save prompts to file
        if dataset_type == 'visa':
            output_filename = f"{sample_name}_prompts.txt"
        else:
            output_filename = f"{sample_name}_{defect_name}_prompts.txt"
            
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for prompt in unique_prompts:
                f.write(prompt + '\n')
        
        print(f"Prompts saved to: {output_path}")
        
        # Also save detailed analysis results
        detailed_output_path = os.path.join(output_dir, f"{output_filename.replace('.txt', '_detailed.json')}")
        
        detailed_results = {
            "sample_name": sample_name,
            "defect_name": defect_name,
            "dataset_type": dataset_type,
            "total_samples": len(matched_pairs),
            "successful_analyses": successful_analyses,
            "original_descriptions_count": len(original_descriptions),
            "unique_prompts": unique_prompts,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(detailed_output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        return unique_prompts


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Diffusion model prompt generator powered by a local vision-language model')
    parser.add_argument('--data_root', required=True, help='Data root directory containing defect images')
    parser.add_argument('--mask_dir', required=True, help='Mask directory')
    parser.add_argument('--output_dir', default='diffusion_prompts', help='Output directory')
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'], default='mvtec', help='Dataset type')
    parser.add_argument('--selected_sample', type=str, default=None, help='Process only the specified sample type')
    parser.add_argument('--selected_defect', type=str, default=None, help='Process only the specified defect type')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to analyze per category')
    parser.add_argument('--max_image_size', type=int, default=512, help='Maximum image dimension (max of width or height), default 512 to reduce VRAM usage')
    parser.add_argument('--api_key', type=str, default=None, help='Deprecated parameter kept for compatibility; not used')
    parser.add_argument('--model_path', type=str, default=None, help='VLM model path or HuggingFace model ID (default: see config.py)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # MVTec dataset info
    mvtec_info = {
        'bottle': ['broken_large', 'broken_small', 'contamination'],
        'cable': ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 'cut_outer_insulation', 'missing_cable','missing_wire','poke_insulation'],
        'capsule': ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
        'carpet': ['color', 'cut', 'hole', 'metal_contamination', 'thread'],
        'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
        'hazelnut': ['crack', 'cut', 'hole', 'print'],
        'leather': ['color', 'cut', 'fold', 'glue', 'poke'],
        'metal_nut': ['bent', 'color', 'scratch'],
        'pill': ['color', 'combined', 'contamination', 'crack', 'faulty_imprint', 'pill_type', 'scratch'],
        'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
        'toothbrush': ['defective'],
        'transistor': ['bent_lead', 'cut_lead', 'damaged_case', 'misplaced'],
        'wood': ['color', 'combined', 'hole', 'liquid', 'scratch'],
        'zipper': ['broken_teeth', 'combined', 'fabric_border', 'fabric_interior', 'split_teeth', 'rough', 'squeezed_teeth'],
    }
    
    # VISA dataset info
    visa_info = {
        # 'cashew': ['anomaly'],
        # 'chewinggum': ['anomaly'],
        # 'fryum': ['anomaly'],
        # 'macaroni1': ['anomaly'],
        # 'macaroni2': ['anomaly'],
        # 'pcb1': ['anomaly'],
        # 'pcb2': ['anomaly'],
        # 'pcb3': ['anomaly'],
        # 'pcb4': ['anomaly'],
        # 'pipe_fryum': ['anomaly'],
        # 'candle': ['anomaly'],
        # 'capsules': ['anomaly'],
        'AeBAD_S': ['ko'],
        'bracket_black': ['ko'],
        'bracket_white': ['ko'],
        'cable': ['ko'],
        'hazelnut': ['ko'],
        'leather': ['ko'],
        'tubes': ['ko'],
        'wood': ['ko'],
    }
    
    # Select dataset info based on dataset type
    if args.dataset == 'mvtec':
        dataset_info = mvtec_info
        print("Using MVTec dataset configuration")
    elif args.dataset == 'visa':
        dataset_info = visa_info
        print("Using VISA dataset configuration")
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")
    
    # Initialize generator
    generator = DiffusionPromptGenerator(
        max_image_size=args.max_image_size,
        model_path=args.model_path,
    )
    print(f"Image size limit: {args.max_image_size} pixels")
    
    # Statistics
    stats = {
        "total_prompts": 0,
        "total_original_descriptions": 0,
        "total_categories": 0,
        "failed_categories": [],
        "successful_categories": []
    }
    
    total_start_time = time.time()
    
    # Process each sample and defect type
    samples_to_process = [args.selected_sample] if args.selected_sample else dataset_info.keys()
    
    for sample_name in samples_to_process:
        if sample_name not in dataset_info:
            print(f"Warning: Unknown sample type {sample_name}")
            continue
            
        defects_to_process = [args.selected_defect] if args.selected_defect else dataset_info[sample_name]
        
        for defect_name in defects_to_process:
            if defect_name not in dataset_info[sample_name]:
                print(f"Warning: Defect type {defect_name} not found in sample {sample_name}")
                continue
                
            stats["total_categories"] += 1
            category_start_time = time.time()
            
            try:
                # Generate prompts
                prompts = generator.generate_prompts_for_dataset(
                    sample_name,
                    defect_name,
                    args.data_root,
                    args.mask_dir,
                    args.output_dir,
                    args.dataset,
                    args.max_samples
                )
                
                if prompts:
                    stats["total_prompts"] += len(prompts)
                    stats["successful_categories"].append(f"{sample_name}/{defect_name}")
                    
                    category_end_time = time.time()
                    print(f"Completed {sample_name}/{defect_name}, generated {len(prompts)} prompts, elapsed {category_end_time - category_start_time:.2f} seconds")
                else:
                    stats["failed_categories"].append(f"{sample_name}/{defect_name}")
                    print(f"Failed to process {sample_name}/{defect_name}")
                    
            except Exception as e:
                print(f"Error processing {sample_name}/{defect_name}: {e}")
                stats["failed_categories"].append(f"{sample_name}/{defect_name}")
    
    total_end_time = time.time()
    
    # Save statistics
    stats["total_time"] = total_end_time - total_start_time
    stats_file = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n========== Prompt generation complete ==========")
    print(f"Total prompts generated: {stats['total_prompts']}")
    print(f"Categories processed: {stats['total_categories']}")
    print(f"Successful: {len(stats['successful_categories'])}")
    print(f"Failed: {len(stats['failed_categories'])}")
    print(f"Total elapsed time: {stats['total_time']:.2f} seconds")
    
    if stats['successful_categories']:
        print("\nSuccessfully processed categories:")
        for category in stats['successful_categories']:
            print(f"  - {category}")
    
    if stats['failed_categories']:
        print("\nFailed categories:")
        for category in stats['failed_categories']:
            print(f"  - {category}")
    
    print("===============================")


if __name__ == "__main__":
    main()

# Usage examples:
# 
# MVTec dataset:
# python prompt_generation.py \
#     --data_root mvtec_train_data \
#     --mask_dir mvtec_train_data \
#     --output_dir prompts_mvtec \
#     --dataset mvtec \
#     --selected_sample transistor \
#     --selected_defect misplaced \
#     --max_samples 5
#
# VISA dataset:
# python prompt_generation.py \
#     --data_root visa_train_data_balanced \
#     --mask_dir visa_train_data_balanced \
#     --output_dir prompts_visa \
#     --dataset visa \
#     --selected_sample pcb1 \
#     --max_samples 10 
# MIST dataset:
# python prompt_generation.py \
#     --data_root datasets/AD/mixdatasets_ref \
#     --mask_dir datasets/AD/mixdatasets_ref \
#     --output_dir prompts_misdatasets \
#     --dataset visa \
#     --max_samples 10 
