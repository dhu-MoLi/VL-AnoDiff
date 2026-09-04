import os
import json
import time
import argparse
import numpy as np
import cv2
import requests
import base64
import random
import glob
from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from scipy.ndimage import gaussian_filter

from config import DEFAULT_MODEL_PATH


class LLMDefectAnalyzer:
    """Industrial defect analyzer powered by a large language model.
    
    Uses LLM reasoning to analyze defect types and recommend suitable mask generation strategies.
    """
    
    def __init__(self, api_key=None, cache_dir="llm_cache", model_path=None):
        """Initialize the analyzer.
        
        Args:
            api_key: No longer used
            cache_dir: Cache directory for storing LLM responses to reduce API calls
            model_path: VLM model path or HuggingFace model ID
        """
        self.cache_dir = cache_dir
        self.cache = {}
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try loading previous analysis results from cache file
        cache_file = os.path.join(cache_dir, "defect_analysis_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Unable to load cache file: {e}")
    
        # Load local model
        print(f"Loading Qwen2.5-VL model: {self.model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load processor
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
    
    def _encode_image(self, image_path):
        """Encode image to base64 string for LLM with image input support"""
        # Use the same encoding method as in API.py
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _build_prompt(self, sample_name, defect_name, visual_evidence=None):
        """Build LLM prompt"""
        prompt_text = f"""
        As an industrial defect analysis expert, please analyze the following industrial product defect and recommend suitable mask generation strategies.

        Product Category: {sample_name}
        Defect Type: {defect_name}
        
        Please analyze this defect based on the following classification system:
        1. Structural defects: Changes in the object's structure/material, such as cracks, scratches, contamination, etc.
        2. Logical defects: Changes in logical relationships like position/orientation/existence, such as misplacement, missing parts, wrong orientation, etc.
        """
        
        # Add visual evidence if available
        if visual_evidence and visual_evidence.get("mask_descriptions"):
            prompt_text += "\nDefect mask feature analysis:\n"
            for desc in visual_evidence["mask_descriptions"]:
                prompt_text += f"- {desc}\n"
        
        # Explicitly list available strategies
        prompt_text += """
        Available mask generation strategies:
        1. elastic_deformation: For structural defects that involve shape changes
        2. texture_modification: For surface texture or pattern changes
        3. edge_enhancement: For edge-related defects
        4. fracture_simulation: For crack or break-like defects
        5. translation: For position-related logical defects
        6. rotation: For orientation-related logical defects
        7. component_removal: For missing component defects
        8. component_addition: For extra component defects

        Please provide the following content:
        1. Defect classification (structural or logical)
        2. Classification reasoning
        3. Recommended mask generation strategies (MUST be one or more of the above strategies)
        4. Suitable physical constraint settings for this defect (such as allowing deformation, maintaining topological structure, etc.)
        5. Defect severity estimation (low/medium/high)
        
        Please return results in JSON format as follows:
        {
            "defect_type": "structural or logical",
            "reasoning": "classification reasoning",
            "recommended_strategies": ["strategy1", "strategy2", ...],  # MUST use exact strategy names from the list above
            "physical_constraints": {
                "allow_deformation": true/false,
                "preserve_topology": true/false,
                "allow_fragmentation": true/false,
                ...
            },
            "severity": "low/medium/high"
        }
        """
        
        return prompt_text
    
    def _call_llm_api(self, prompt, image_paths=None):
        """Run inference with local model"""
        try:
            if image_paths:
                # Process image input
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *[{"type": "image", "image": path} for path in image_paths]
                        ]
                    }
                ]
            else:
                # Text-only input
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]
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
            
            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            print(f"Model inference failed: {e}")
            return None
    
    def _describe_mask(self, mask):
        """Describe mask characteristics"""
        if mask is None or np.sum(mask > 0) == 0:
            return None
            
        height, width = mask.shape
        
        # Calculate basic features
        area = np.sum(mask > 0)
        area_ratio = area / (height * width)
        
        # Calculate bounding box
        y_indices, x_indices = np.where(mask > 0)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        
        # Calculate shape complexity
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0
        for contour in contours:
            perimeter += cv2.arcLength(contour, True)
        
        complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 1.0
        
        # Generate description
        description = f"Area ratio: {area_ratio:.3f}, Aspect ratio: {aspect_ratio:.2f}, Complexity: {complexity:.2f}"
        
        if area_ratio < 0.05:
            description += ", Small defect"
        elif area_ratio > 0.3:
            description += ", Large defect"
        else:
            description += ", Medium defect"
            
        if complexity > 3.0:
            description += ", Complex shape"
        elif complexity < 1.5:
            description += ", Simple shape"
        else:
            description += ", Moderate complexity"
            
        return description
    
    def _prepare_visual_evidence(self, mask_paths, defect_images, max_samples=3):
        """Prepare visual evidence for LLM analysis"""
        # Select samples
        if not mask_paths or len(mask_paths) == 0:
            return None
            
        mask_samples = np.random.choice(mask_paths, min(max_samples, len(mask_paths)), replace=False)
        image_samples = []
        if defect_images and len(defect_images) > 0:
            image_samples = np.random.choice(defect_images, min(max_samples, len(defect_images)), replace=False)
        
        visual_evidence = {
            "mask_descriptions": [],
            "image_descriptions": []
        }
        
        # Analyze mask samples
        for mask_path in mask_samples:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # Compute basic features
            description = self._describe_mask(mask)
            if description:
                visual_evidence["mask_descriptions"].append(description)
        
        return visual_evidence
    
    def _parse_llm_response(self, response_content):
        """Parse LLM response"""
        if not response_content:
            return None
            
        try:
            # Try to extract JSON from response
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_content, re.DOTALL)
            
            for json_str in json_matches:
                try:
                    result = json.loads(json_str)
                    if "defect_type" in result:
                        return result
                except json.JSONDecodeError:
                    continue
            
            # If JSON parsing fails, try to parse based on keywords
            result = {
                "defect_type": "structural",
                "reasoning": "Default classification",
                "recommended_strategies": ["elastic_deformation", "texture_modification"],
                "physical_constraints": {
                    "allow_deformation": True,
                    "preserve_topology": False,
                    "allow_fragmentation": True
                },
                "severity": "medium"
            }
            
            if "logical" in response_content.lower():
                result["defect_type"] = "logical"
                result["recommended_strategies"] = ["translation", "rotation"]
                result["physical_constraints"]["allow_deformation"] = False
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
    
    def _fallback_analysis(self, sample_name, defect_name):
        """Fallback analysis when LLM is unavailable"""
        # Simple heuristic classification based on defect name
        logical_keywords = ["missing", "misplaced", "wrong", "position", "orientation", "swap", "bent_lead", "cut_lead"]
        structural_keywords = ["crack", "scratch", "contamination", "hole", "cut", "broken", "poke", "squeeze"]
        
        defect_lower = defect_name.lower()
        
        if any(keyword in defect_lower for keyword in logical_keywords):
            defect_type = "logical"
            strategies = ["translation", "rotation", "component_removal"]
            constraints = {
                "allow_deformation": False,
                "preserve_topology": True,
                "allow_fragmentation": False
            }
        else:
            defect_type = "structural"
            strategies = ["elastic_deformation", "texture_modification", "fracture_simulation"]
            constraints = {
                "allow_deformation": True,
                "preserve_topology": False,
                "allow_fragmentation": True
            }
        
        return {
            "defect_type": defect_type,
            "reasoning": f"Heuristic classification based on defect name: {defect_name}",
            "recommended_strategies": strategies,
            "physical_constraints": constraints,
            "severity": "medium"
        }
    
    def _save_cache(self):
        """Save cache to file"""
        cache_file = os.path.join(self.cache_dir, "defect_analysis_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Unable to save cache: {e}")
    
    def analyze_defect(self, sample_name, defect_name, mask_paths=None, defect_images=None):
        """Analyze defect type and recommend generation strategies"""
        # Check cache
        cache_key = f"{sample_name}_{defect_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Prepare visual evidence
        visual_evidence = self._prepare_visual_evidence(mask_paths or [], defect_images or [])
        
        # Build prompt
        prompt = self._build_prompt(sample_name, defect_name, visual_evidence)
        
        # Call LLM API
        response = self._call_llm_api(prompt)
        
        # Parse response
        analysis_result = self._parse_llm_response(response)
        
        # Use fallback if LLM analysis fails
        if not analysis_result:
            analysis_result = self._fallback_analysis(sample_name, defect_name)
        
        # Cache result
        if analysis_result:
            self.cache[cache_key] = analysis_result
            self._save_cache()
        
        return analysis_result
    
    def analyze_defect_with_images(self, sample_name, defect_name, image_paths):
        """Analyze defect type with images and recommend generation strategies.
        
        Args:
            sample_name: Sample category name
            defect_name: Defect type name
            image_paths: List of defect image file paths
            
        Returns:
            dict: Analysis result including defect type, generation strategies, etc.
        """
        # Check cache
        cache_key = f"{sample_name}_{defect_name}_image"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not image_paths or len(image_paths) == 0:
            return self._fallback_analysis(sample_name, defect_name)
        
        # Select up to 3 images for analysis
        # selected_images = image_paths[:min(3, len(image_paths))]
        selected_images = image_paths
        
        # Build prompt text
        prompt_text = self._build_prompt(sample_name, defect_name)
        
        # Run inference with local model
        response_content = self._call_llm_api(prompt_text, selected_images)
        
        # Parse response
        analysis_result = self._parse_llm_response(response_content)
        
        # Cache result
        if analysis_result:
            self.cache[cache_key] = analysis_result
            self._save_cache()
        
        return analysis_result or self._fallback_analysis(sample_name, defect_name)


class LLMMaskGenerator:
    """Generate masks using strategies recommended by the LLM"""
    
    def __init__(self, image_size=(256, 256), api_key=None, dataset_type='mvtec', model_path=None):
        self.image_size = image_size
        self.dataset_type = dataset_type  # Dataset type identifier
        self.analyzer = LLMDefectAnalyzer(model_path=model_path)
        
        # Strategy dictionary
        self.strategies = {
            # Structural defect strategies
            "elastic_deformation": self._apply_elastic_deformation,
            "texture_modification": self._modify_texture,
            "edge_enhancement": self._enhance_edges,
            "fracture_simulation": self._simulate_fracture,
            
            # Logical defect strategies
            "translation": self._apply_translation,
            "rotation": self._apply_rotation,
            "component_removal": self._remove_component,
            "component_addition": self._add_component,
        }
    
    def _get_file_patterns(self, sample_name, defect_name, data_root, mask_dir):
        """Return file patterns based on dataset type"""
        if self.dataset_type == 'mvtec':
            # MVTec dataset file structure
            mask_pattern = os.path.join(mask_dir, sample_name, f"{defect_name}_mask", "*.png")
            img_pattern = os.path.join(data_root, sample_name, defect_name, "*.png")
        elif self.dataset_type == 'visa':
            # VISA dataset file structure
            mask_pattern = os.path.join(mask_dir, sample_name, "ko_mask", "*")
            img_pattern = os.path.join(data_root, sample_name, "ko", "*")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
        
        return mask_pattern, img_pattern
    
    def generate_masks(self, sample_name, defect_name, data_root, mask_dir, output_dir, 
                      num_simple=3, num_complex=3):
        """Generate masks (supports multiple datasets)"""
        if self.dataset_type == 'visa':
            print(f"\n==== Generating LLM-guided masks for VISA dataset {sample_name} ====")
        else:
            print(f"\n==== Generating LLM-guided masks for {sample_name}/{defect_name} ====")
        
        # Create output directories
        if self.dataset_type == 'visa':
            # VISA dataset uses sample name as main directory
            simple_dir = os.path.join(output_dir, sample_name, "simple")
            complex_dir = os.path.join(output_dir, sample_name, "complex")
            merged_dir = os.path.join(output_dir, sample_name)  # Merged directory
        else:
            # MVTec dataset keeps original structure
            simple_dir = os.path.join(output_dir, sample_name, defect_name, "simple")
            complex_dir = os.path.join(output_dir, sample_name, defect_name, "complex")
            merged_dir = os.path.join(output_dir, sample_name)  # Merged directory
        
        os.makedirs(simple_dir, exist_ok=True)
        os.makedirs(complex_dir, exist_ok=True)
        os.makedirs(merged_dir, exist_ok=True)
        
        # Get file patterns based on dataset type
        mask_pattern, img_pattern = self._get_file_patterns(sample_name, defect_name, data_root, mask_dir)
        
        # Collect reference masks and defect images
        reference_masks = glob.glob(mask_pattern)
        defect_images = glob.glob(img_pattern)
        
        print(f"Found {len(reference_masks)} reference mask files")
        print(f"Found {len(defect_images)} defect image files")
        
        if len(reference_masks) == 0:
            print(f"Warning: No reference mask files found, pattern: {mask_pattern}")
        if len(defect_images) == 0:
            print(f"Warning: No defect image files found, pattern: {img_pattern}")
        
        # Use LLM to analyze defect type
        # For VISA dataset, use generic anomaly analysis
        if self.dataset_type == 'visa':
            # Provide more generic analysis prompt for VISA dataset
            analysis_defect_name = f"{sample_name} anomaly"
        else:
            analysis_defect_name = defect_name
        
        # Prefer image-based analysis when defect images are available
        if defect_images and len(defect_images) > 0:
            analysis = self.analyzer.analyze_defect_with_images(sample_name, analysis_defect_name, defect_images)
        else:
            analysis = self.analyzer.analyze_defect(sample_name, analysis_defect_name, reference_masks, defect_images)
        
        if not analysis:
            print(f"Warning: LLM analysis failed, using default settings")
            # Provide more targeted default settings for different VISA sample types
            if self.dataset_type == 'visa':
                if 'pcb' in sample_name.lower():
                    # PCB types tend toward logical defects
                    defect_type = "logical"
                    strategies = ["component_removal", "translation", "rotation"]
                    constraints = {
                        "allow_deformation": False,
                        "preserve_topology": True,
                        "allow_fragmentation": False
                    }
                else:
                    # Food types tend toward structural defects
                    defect_type = "structural"
                    strategies = ["elastic_deformation", "texture_modification"]
                    constraints = {
                        "allow_deformation": True,
                        "preserve_topology": False,
                        "allow_fragmentation": True
                    }
            else:
                defect_type = "structural"  # Default to structural defect
                strategies = ["elastic_deformation", "texture_modification"]
                constraints = {
                    "allow_deformation": True,
                    "preserve_topology": False,
                    "allow_fragmentation": True
                }
        else:
            defect_type = analysis.get("defect_type", "structural")
            strategies = analysis.get("recommended_strategies", [])
            constraints = analysis.get("physical_constraints", {})
            
        print(f"LLM analysis result: defect type = {defect_type}")
        print(f"Recommended strategies: {strategies}")
        print(f"Physical constraints: {constraints}")
        
        # Analyze physical constraints
        physical_constraints = self._analyze_physical_constraints(
            sample_name, analysis_defect_name, reference_masks, defect_images
        )
        
        # Generate simple masks
        simple_count = 0
        simple_masks = []  # Store paths of simple masks
        for i in range(num_simple):
            max_attempts = 100  # Maximum retry count
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # Select a reference mask
                    ref_mask = random.choice(reference_masks) if reference_masks else None
                    
                    # Select one from recommended strategies
                    if strategies:
                        strategy_name = random.choice(strategies)
                    else:
                        # Select default strategy based on defect type
                        strategy_name = "elastic_deformation" if defect_type == "structural" else "translation"
                    
                    # Generate mask
                    mask = self._generate_mask(
                        strategy_name, ref_mask, constraints, physical_constraints, simple=True
                    )
                    
                    # Check whether generated mask is valid
                    if mask is not None and np.sum(mask > 0) > 0:
                        output_path = os.path.join(simple_dir, f"{simple_count:03d}.png")
                        cv2.imwrite(output_path, mask)
                        simple_masks.append(output_path)
                        simple_count += 1
                        print(f"Generated simple mask: {output_path}, strategy: {strategy_name}")
                        break  # Successfully generated, exit retry loop
                    else:
                        attempt += 1
                        print(f"Simple mask generation failed, retry {attempt}/{max_attempts}")
                        
                except Exception as e:
                    attempt += 1
                    print(f"Error generating simple mask: {e}, retry {attempt}/{max_attempts}")
            
            if attempt >= max_attempts:
                print(f"Warning: Simple mask generation failed, max retries reached")
        
        # Generate complex masks
        complex_count = 0
        complex_masks = []  # Store paths of complex masks
        for i in range(num_complex):
            max_attempts = 100  # Maximum retry count
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # Select a reference mask
                    ref_mask = random.choice(reference_masks) if reference_masks else None
                    
                    # Select one from recommended strategies
                    if strategies:
                        strategy_name = random.choice(strategies)
                    else:
                        # Select default strategy based on defect type
                        strategy_name = "fracture_simulation" if defect_type == "structural" else "component_removal"
                    
                    # Generate mask
                    mask = self._generate_mask(
                        strategy_name, ref_mask, constraints, physical_constraints, simple=False
                    )
                    
                    # Check whether generated mask is valid
                    if mask is not None and np.sum(mask > 0) > 0:
                        output_path = os.path.join(complex_dir, f"{complex_count:03d}.png")
                        cv2.imwrite(output_path, mask)
                        complex_masks.append(output_path)
                        complex_count += 1
                        print(f"Generated complex mask: {output_path}, strategy: {strategy_name}")
                        break  # Successfully generated, exit retry loop
                    else:
                        attempt += 1
                        print(f"Complex mask generation failed, retry {attempt}/{max_attempts}")
                        
                except Exception as e:
                    attempt += 1
                    print(f"Error generating complex mask: {e}, retry {attempt}/{max_attempts}")
            
            if attempt >= max_attempts:
                print(f"Warning: Complex mask generation failed, max retries reached")
        
        # Merge simple and complex masks into unified directory
        print(f"\nMerging masks into {merged_dir}")
        merged_count = 0
        
        # Copy simple masks
        for mask_path in simple_masks:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    merged_path = os.path.join(merged_dir, f"{merged_count:03d}.png")
                    cv2.imwrite(merged_path, mask)
                    merged_count += 1
                    print(f"Merged simple mask: {merged_path}")
            except Exception as e:
                print(f"Error merging simple mask: {e}")
        
        # Copy complex masks
        for mask_path in complex_masks:
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    merged_path = os.path.join(merged_dir, f"{merged_count:03d}.png")
                    cv2.imwrite(merged_path, mask)
                    merged_count += 1
                    print(f"Merged complex mask: {merged_path}")
            except Exception as e:
                print(f"Error merging complex mask: {e}")
        
        print(f"Merge complete, generated {merged_count} mask files in {merged_dir}")
        
        return simple_count, complex_count
    
    def _preprocess_reference_mask(self, ref_mask, simple=True):
        """Preprocess reference mask; upscale if area is too small"""
        if ref_mask is None:
            return None
            
        height, width = ref_mask.shape
        total_pixels = height * width
        
        # Compute area of non-zero pixels in mask
        mask_area = np.sum(ref_mask > 0)
        area_ratio = mask_area / total_pixels
        
        # Set minimum area threshold (consistent with _final_mask_cleanup)
        min_area_threshold = max(50, total_pixels * 0.005)  # At least 50 pixels or 0.5% of total area
        min_area_ratio = min_area_threshold / total_pixels
        
        # Upscale if area is too small
        if mask_area < min_area_threshold:
            print(f"Reference mask area too small ({mask_area} < {min_area_threshold}), upscaling")
            
            # Compute required scale factor (typically 2/3)
            # target_area = min_area_threshold * (2 if simple else 3)  # 2x for simple mode, 3x for complex mode
            target_area = min_area_threshold * (10 if simple else 20)
            scale_factor = np.sqrt(target_area / mask_area)
            
            # Limit max scale factor to avoid over-upscaling (typically 5x)
            scale_factor = min(scale_factor, 20.0)
            
            # Compute new dimensions
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Upscale mask
            enlarged_mask = cv2.resize(ref_mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Ensure binary image
            _, enlarged_mask = cv2.threshold(enlarged_mask, 127, 255, cv2.THRESH_BINARY)
            
            # If upscaled mask is still too small, expand further with morphology
            if np.sum(enlarged_mask > 0) < min_area_threshold:
                # Use dilation to expand mask
                kernel_size = max(3, int(scale_factor * 2))
                # Ensure kernel_size is odd (required for some morphological ops)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                enlarged_mask = cv2.dilate(enlarged_mask, kernel, iterations=1)
                
                # Use closing to fill possible holes
                enlarged_mask = cv2.morphologyEx(enlarged_mask, cv2.MORPH_CLOSE, 
                                               np.ones((3, 3), np.uint8), iterations=1)
            
            print(f"Area after upscaling: {np.sum(enlarged_mask > 0)} (target: {min_area_threshold})")
            return enlarged_mask
        
        return ref_mask
    
    def _generate_mask(self, strategy_name, ref_mask_path, semantic_constraints, 
                     physical_constraints, simple=True):
        """Generate mask using specified strategy"""
        width, height = self.image_size
        
        # Strategy name mapping (handles English names returned by model)
        strategy_mapping = {
            "elastic deformation": "elastic_deformation",
            "adding noise": "texture_modification",
            "edge enhancement": "edge_enhancement",
            "fracture simulation": "fracture_simulation",
            "translation": "translation",
            "rotation": "rotation",
            "component removal": "component_removal",
            "component addition": "component_addition"
        }
        
        # Map strategy name
        mapped_strategy = strategy_mapping.get(strategy_name, strategy_name)
        
        # Use reference mask if available
        if ref_mask_path:
            ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
            if ref_mask is not None:
                ref_mask = cv2.resize(ref_mask, self.image_size)
                
                # Check reference mask area; preprocess and upscale if too small
                ref_mask = self._preprocess_reference_mask(ref_mask, simple)
                if np.sum(ref_mask > 0) == 0:
                    print("Warning: Reference mask is all black after upscaling, skipping this mask")
                    return None
                ref_mask = self._apply_rotation(ref_mask, semantic_constraints, simple)
                if np.sum(ref_mask > 0) == 0:
                    print("Warning: Reference mask is all black after rotation, skipping this mask")
                    return None

                # Look up strategy function
                strategy_func = self.strategies.get(mapped_strategy)
                print(strategy_func)
                if strategy_func:
                    # Apply strategy
                    mask = strategy_func(ref_mask, semantic_constraints, simple)
                    mask_area = np.sum(mask > 0)
                    print(f"Mask area after strategy: {mask_area}")
                    if mask_area == 0:
                        print("Warning: Mask is all black after strategy, skipping this mask")
                        return None
                else:
                    # Default to rotation
                    mask = self._apply_rotation(ref_mask, semantic_constraints, simple)
                
                # Apply final cleanup
                mask = self._final_mask_cleanup(mask)
                mask_area = np.sum(mask > 0)
                print(f"Mask area after cleanup: {mask_area}")
                # if mask is None or np.sum(mask > 0) == 0:
                #     # If mask is all black, return None for caller to regenerate
                #     print(f"Warning: Generated mask is all black, will regenerate")
                #     return None
                # Apply physical constraints
                mask = self._apply_physical_constraints(mask, physical_constraints)
                mask_area = np.sum(mask > 0)
                print(f"Mask area after physical constraints: {mask_area}")
                
                # Final safety check: ensure mask is not all black
                if mask is None or np.sum(mask > 0) == 0:
                    # If mask is all black, return None for caller to regenerate
                    print(f"Warning: Generated mask is all black, will regenerate")
                    return None
                
                return mask
        
        # Generate mask from scratch if no reference mask or unreadable
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Generate base mask based on defect type
        if mapped_strategy in ["elastic_deformation", "texture_modification", "edge_enhancement", "fracture_simulation"]:
            # Structural defect: generate irregular shape
            num_vertices = random.randint(3, 8) if simple else random.randint(5, 15)  # Vertex count logic
            
            # Generate irregular polygon using polar coordinates
            vertices = []
            center_x = random.randint(width // 4, width * 3 // 4)
            center_y = random.randint(height // 4, height * 3 // 4)
            
            for i in range(num_vertices):
                # Generate vertices using polar coordinates
                angle = 2 * np.pi * i / num_vertices + random.uniform(-0.3, 0.3)  # Add random offset
                r = random.randint(20, 80) if simple else random.randint(15, 100)  # Radius range
                
                # Convert to Cartesian coordinates
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                
                # Clamp to image bounds
                x = max(0, min(width-1, x))
                y = max(0, min(height-1, y))
                
                vertices.append((x, y))
            
            # Draw polygon
            vertices = np.array(vertices, dtype=np.int32)
            cv2.fillPoly(mask, [vertices], 255)
        
        else:  # Logical defect: generate regular shape
            shape_type = random.choice(["rectangle", "circle", "ellipse"])#
            
            if shape_type == "rectangle":
                w = random.randint(30, 80)
                h = random.randint(30, 80)
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
            elif shape_type == "circle":
                radius = random.randint(20, 60)
                x = random.randint(radius, width - radius)
                y = random.randint(radius, height - radius)
                cv2.circle(mask, (x, y), radius, 255, -1)
                
            elif shape_type == "ellipse":
                center = (random.randint(width // 4, width * 3 // 4), 
                          random.randint(height // 4, height * 3 // 4))
                axes = (random.randint(20, 60), random.randint(15, 45))
                angle = random.randint(0, 180)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        # Apply specified strategy
        strategy_func = self.strategies.get(mapped_strategy)
        if strategy_func:
            mask = strategy_func(mask, semantic_constraints, simple)
            mask = self._final_mask_cleanup(mask)
        # Apply physical constraints
        mask = self._apply_physical_constraints(mask, physical_constraints)
        
        return mask
    
    # Strategy implementation functions below...
    def _apply_elastic_deformation(self, mask, constraints, simple=True):
        """Apply elastic deformation"""
        if not constraints.get("allow_deformation", True):
            return mask
            
        height, width = mask.shape
        
        # Set random deformation strength
        if simple:
            alpha = random.uniform(8, 20)   # Simple mode: smaller deformation strength
            sigma = random.uniform(2, 4)    # Smaller smoothing parameter
        else:
            alpha = random.uniform(15, 35)  # Complex mode: moderate deformation strength
            sigma = random.uniform(3, 6)    # Moderate smoothing parameter
        
        # Create random displacement field
        dx = np.random.rand(height, width) * 2 - 1
        dy = np.random.rand(height, width) * 2 - 1
        
        # Smooth displacement field
        dx = gaussian_filter(dx, sigma) * alpha
        dy = gaussian_filter(dy, sigma) * alpha
        
        # Create grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Apply displacement
        new_y = y + dy
        new_x = x + dx
        
        # Clamp deformed coordinates to image bounds
        new_y = np.clip(new_y, 0, height - 1)
        new_x = np.clip(new_x, 0, width - 1)
        
        # Convert to integer indices
        indices_y = new_y.astype(np.int32)
        indices_x = new_x.astype(np.int32)
        
        # Apply deformation using efficient method
        distorted = np.zeros_like(mask)
        
        # Create valid index mask
        valid_mask = (indices_y >= 0) & (indices_y < height) & (indices_x >= 0) & (indices_x < width)
        
        # Deform only valid indices
        if np.any(valid_mask):
            distorted[valid_mask] = mask[indices_y[valid_mask], indices_x[valid_mask]]
        
        # Ensure result is binary image
        _, distorted = cv2.threshold(distorted, 127, 255, cv2.THRESH_BINARY)
        
        # Quality check: ensure deformed mask is still valid
        original_area = np.sum(mask > 0)
        deformed_area = np.sum(distorted > 0)
        
        # If deformation changes area too much, reduce strength
        if original_area > 0:
            area_ratio = deformed_area / original_area
            if area_ratio < 0.3 or area_ratio > 3.0:  # Area change too large
                # Use milder deformation
                alpha = alpha * 0.5
                dx = gaussian_filter(np.random.rand(height, width) * 2 - 1, sigma) * alpha
                dy = gaussian_filter(np.random.rand(height, width) * 2 - 1, sigma) * alpha
                
                new_y = np.clip(y + dy, 0, height - 1)
                new_x = np.clip(x + dx, 0, width - 1)
                
                indices_y = new_y.astype(np.int32)
                indices_x = new_x.astype(np.int32)
                
                distorted = np.zeros_like(mask)
                valid_mask = (indices_y >= 0) & (indices_y < height) & (indices_x >= 0) & (indices_x < width)
                if np.any(valid_mask):
                    distorted[valid_mask] = mask[indices_y[valid_mask], indices_x[valid_mask]]
                _, distorted = cv2.threshold(distorted, 127, 255, cv2.THRESH_BINARY)
        
        # Optional: light morphology on deformed mask for continuity
        if np.sum(distorted > 0) > 0:
            # Use small kernel closing to fill possible holes
            kernel = np.ones((3, 3), np.uint8)
            distorted = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return distorted

    def _modify_texture(self, mask, constraints, simple=True):
        """Modify texture"""
        if not constraints.get("texture_focus", True):
            return mask
            
        # Basic processing: add noise
        noise_scale = random.uniform(0, 0.3) if simple else random.uniform(0.3, 0.9)  # Use random.uniform
        
        # Create noise
        height, width = mask.shape
        noise = np.random.rand(height, width) * 255
        noise = (noise > 200).astype(np.uint8) * 255
        
        # Apply Gaussian blur for natural-looking noise
        # kernel_size = 3 if simple else 5
        kernel_size = random.randint(3, 15) if simple else random.randint(5, 20)  # Reasonable kernel size range
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)
        
        # Add noise only within mask region
        noise_mask = cv2.bitwise_and(noise, mask)
        
        # Combine original mask and noise
        result = cv2.bitwise_or(mask, noise_mask)
        
        # For complex masks, add more texture
        if not simple:
            # Create additional texture layer
            texture_layer = np.zeros((height, width), dtype=np.uint8)
            
            # Add random spots
            num_spots = random.randint(3, 30)
            for _ in range(num_spots):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                radius = random.randint(2, 8)
                cv2.circle(texture_layer, (x, y), radius, 255, -1)
            
            # Apply texture only within mask region
            texture_mask = cv2.bitwise_and(texture_layer, mask)
            
            # Merge results
            result = cv2.bitwise_or(result, texture_mask)
        # Quality check: ensure result is not all black
        if np.sum(result > 0) == 0:
            print("Warning: Texture modification failed, returning original mask")
            return mask
        
        return result

    def _enhance_edges(self, mask, constraints, simple=True):
        """Enhance edges"""
        if not constraints.get("edge_focus", True):
            return mask
            
        # Find edges
        edges = cv2.Canny(mask, 100, 200)
        
        # Expand edges
        kernel_size = random.randint(3, 15) if simple else random.randint(5, 20)  # Reasonable kernel size range
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Merge original mask and edges
        result = cv2.bitwise_or(mask, dilated_edges)
        
        # For complex masks, add more edge detail
        if not simple:
            # Make edges more irregular
            noise = np.random.rand(*edges.shape) * 255
            noise = (noise > 240).astype(np.uint8) * 255
            noise_edges = cv2.bitwise_and(noise, dilated_edges)
            
            # Expand noisy edges
            noisy_dilated = cv2.dilate(noise_edges, kernel, iterations=1)
            
            # Merge results
            result = cv2.bitwise_or(result, noisy_dilated)
        
        return result

    def _simulate_fracture(self, mask, constraints, simple=True):
        """Simulate fracture"""
        if not constraints.get("allow_fragmentation", True):
            return mask
            
        height, width = mask.shape
        
        # Find mask region
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return mask
            
        # Compute mask center
        center_y = int(np.mean(y_indices))
        center_x = int(np.mean(x_indices))
        
        # Compute mask bounding box
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        mask_width = max_x - min_x + 1
        mask_height = max_y - min_y + 1
        
        # Create random direction for fracture
        angle = np.random.uniform(0, 2 * np.pi)
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Create fracture line
        fracture_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Improved fracture line generation: ensure line crosses mask region
        line_length = max(mask_width, mask_height) * 1.5  # Based on mask size
        
        # Compute fracture line start/end through mask center
        start_x = int(center_x - dx * line_length / 2)
        start_y = int(center_y - dy * line_length / 2)
        end_x = int(center_x + dx * line_length / 2)
        end_y = int(center_y + dy * line_length / 2)
        
        # Clamp start/end points to image bounds
        start_x = max(0, min(width-1, start_x))
        start_y = max(0, min(height-1, start_y))
        end_x = max(0, min(width-1, end_x))
        end_y = max(0, min(height-1, end_y))
        
        # Draw main fracture line with thickness for visibility
        thickness = random.randint(2, 6) if simple else random.randint(3, 12)
        cv2.line(fracture_mask, (start_x, start_y), (end_x, end_y), 255, thickness)
        
        # For complex masks, add branch fractures
        if not simple:
            num_branches = random.randint(1, 3)  # Reduce branch count
            for _ in range(num_branches):
                # Pick random point on fracture line
                t = np.random.uniform(0.2, 0.8)  # Avoid endpoints
                branch_x = int(start_x + t * (end_x - start_x))
                branch_y = int(start_y + t * (end_y - start_y))
                
                # Clamp branch point to image bounds
                branch_x = max(0, min(width-1, branch_x))
                branch_y = max(0, min(height-1, branch_y))
                
                # Random branch angle
                branch_angle = angle + np.random.uniform(-np.pi/6, np.pi/6)
                branch_dx = np.cos(branch_angle)
                branch_dy = np.sin(branch_angle)
                
                # Branch length
                branch_length = np.random.uniform(0.2, 0.5) * line_length
                branch_end_x = int(branch_x + branch_dx * branch_length)
                branch_end_y = int(branch_y + branch_dy * branch_length)
                
                # Clamp branch end point to image bounds
                branch_end_x = max(0, min(width-1, branch_end_x))
                branch_end_y = max(0, min(height-1, branch_end_y))
                
                # Draw branch
                cv2.line(fracture_mask, (branch_x, branch_y), 
                         (branch_end_x, branch_end_y), 255, 
                         random.randint(1, 3))
        
        # Intersect with original mask
        fracture_mask = cv2.bitwise_and(fracture_mask, mask)
        
        # Check whether fracture is valid
        fracture_area = np.sum(fracture_mask > 0)
        if fracture_area == 0:
            # If fracture invalid, create simpler fracture
            fracture_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Create cross-shaped fracture at mask center
            h_start_x = max(0, min_x - 10)
            h_end_x = min(width-1, max_x + 10)
            cv2.line(fracture_mask, (h_start_x, center_y), (h_end_x, center_y), 255, thickness)
            
            v_start_y = max(0, min_y - 10)
            v_end_y = min(height-1, max_y + 10)
            cv2.line(fracture_mask, (center_x, v_start_y), (center_x, v_end_y), 255, thickness)
            
            # Intersect with original mask again
            fracture_mask = cv2.bitwise_and(fracture_mask, mask)
        
        # Expand fracture region
        kernel_size = random.randint(3, 7) if simple else random.randint(5, 9)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        fracture_mask = cv2.dilate(fracture_mask, kernel, iterations=1)
        
        # Final mask: original mask plus expanded fracture
        result = cv2.bitwise_or(mask, fracture_mask)
        
        # Quality check: ensure result is not all black
        if np.sum(result > 0) == 0:
            print("Warning: Fracture simulation failed, returning original mask")
            return mask
        
        return result

    def _apply_translation(self, mask, constraints, simple=True):
        """Apply translation"""
        height, width = mask.shape
        
        # Determine translation range
        max_tx = int(width * 0.5)
        max_ty = int(height * 0.5)
        
        # Random translation offset
        tx = random.randint(-max_tx, max_tx)
        ty = random.randint(-max_ty, max_ty)
        
        # Ensure minimum translation in at least one direction
        if simple and abs(tx) < width * 0.05 and abs(ty) < height * 0.05:
            if random.choice([True, False]):
                tx = int(width * 0.05) * (1 if tx >= 0 else -1)
            else:
                ty = int(height * 0.05) * (1 if ty >= 0 else -1)
        
        # Create transformation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        translated = cv2.warpAffine(mask, M, (width, height))
        
        return translated

    def _apply_rotation(self, mask, constraints, simple=True):
        """Apply rotation"""
        if not constraints.get("allow_rotation", True):
            return mask
            
        height, width = mask.shape
        center = (width // 2, height // 2)
        
        # Random rotation angle
        max_angle = 180 if simple else 90
        angle = random.uniform(-max_angle, max_angle)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(mask, M, (width, height))
        
        return rotated

    def _remove_component(self, mask, constraints, simple=True):
        """Remove component (for logical defects)"""
        if not constraints.get("allow_fragmentation", False):
            return mask
            
        height, width = mask.shape
        
        # Compute connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # If only one region (besides background), handle differently
        if num_labels <= 2:
            # Simple removal: cut from edge
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                
                # Randomly select an edge for cutting
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                
                if edge == 'top':
                    cut_height = int((max_y - min_y) * random.uniform(0.2, 0.4))
                    mask[min_y:min_y+cut_height, min_x:max_x] = 0
                elif edge == 'bottom':
                    cut_height = int((max_y - min_y) * random.uniform(0.2, 0.4))
                    mask[max_y-cut_height:max_y, min_x:max_x] = 0
                elif edge == 'left':
                    cut_width = int((max_x - min_x) * random.uniform(0.2, 0.4))
                    mask[min_y:max_y, min_x:min_x+cut_width] = 0
                elif edge == 'right':
                    cut_width = int((max_x - min_x) * random.uniform(0.2, 0.4))
                    mask[min_y:max_y, max_x-cut_width:max_x] = 0
        else:
            # Multiple connected regions, can remove one
            # Exclude background (label 0)
            component_sizes = []
            for i in range(1, num_labels):
                component_sizes.append((i, stats[i, cv2.CC_STAT_AREA]))
            
            # Sort components by area
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Remove one component (not the largest)
            if len(component_sizes) > 1:
                if simple:
                    # Remove one of the smaller components
                    remove_idx = random.choice(component_sizes[1:])
                    mask[labels == remove_idx[0]] = 0
                else:
                    # Remove multiple smaller components
                    num_to_remove = min(random.randint(1, 3), len(component_sizes) - 1)
                    for i in range(num_to_remove):
                        remove_idx = component_sizes[i + 1][0]
                        mask[labels == remove_idx] = 0
        
        return mask

    def _add_component(self, mask, constraints, simple=True):
        """Add component (for logical defects)"""
        height, width = mask.shape
        
        # Compute mask bounding box
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return mask
            
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        
        # Create new component
        component_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Decide new component type
        if simple:
            # Simple shapes: circle or rectangle
            shape_type = random.choice(['circle', 'rectangle'])
            
            if shape_type == 'circle':
                radius = random.randint(10, 30)
                
                # Set center near original mask
                offset_range = 30
                center_x = random.randint(max(0, min_x - offset_range), min(width - 1, max_x + offset_range))
                center_y = random.randint(max(0, min_y - offset_range), min(height - 1, max_y + offset_range))
                
                # Draw circle
                cv2.circle(component_mask, (center_x, center_y), radius, 255, -1)
                
            elif shape_type == 'rectangle':
                # Set rectangle dimensions
                rect_width = random.randint(30, 50)
                rect_height = random.randint(30, 50)
                
                # Set top-left position
                offset_range = 30
                rect_x = random.randint(max(0, min_x - offset_range), min(width - rect_width, max_x + offset_range))
                rect_y = random.randint(max(0, min_y - offset_range), min(height - rect_height, max_y + offset_range))
                
                # Draw rectangle
                cv2.rectangle(component_mask, (rect_x, rect_y), 
                             (rect_x + rect_width, rect_y + rect_height), 
                             255, -1)
        else:
            # Complex shape: deformed version of original shape
            # Compute connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            if num_labels > 1:
                # Randomly select one connected component
                component_label = random.randint(1, num_labels - 1)
                component = (labels == component_label).astype(np.uint8) * 255
                
                # Get component centroid
                component_y, component_x = np.where(component > 0)
                if len(component_y) > 0:
                    center_y = int(np.mean(component_y))
                    center_x = int(np.mean(component_x))
                    
                    # Apply random transforms
                    # 1. Scale
                    scale_factor = random.uniform(0.7, 1.3)
                    
                    # 2. Rotate
                    angle = random.uniform(-30, 30)
                    
                    # Create transformation matrix
                    M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale_factor)
                    
                    # 3. Translate
                    tx = random.randint(-30, 30)
                    ty = random.randint(-30, 30)
                    M[0, 2] += tx
                    M[1, 2] += ty
                    
                    # Apply transform
                    component_mask = cv2.warpAffine(component, M, (width, height))
        
        # Ensure new component overlaps or separates appropriately from original mask
        if constraints.get("preserve_topology", True):
            # Ensure no overlap
            overlap = cv2.bitwise_and(mask, component_mask)
            if np.sum(overlap > 0) > 0:
                # If overlap exists, shift new component
                # Find non-overlapping region
                safe_distance = 10
                
                # Try multiple directions
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1), 
                             (1, 1), (-1, -1), (1, -1), (-1, 1)]
                
                for dx, dy in directions:
                    # Create translation matrix
                    M = np.float32([[1, 0, dx * safe_distance], 
                                    [0, 1, dy * safe_distance]])
                    
                    # Translate new component
                    shifted = cv2.warpAffine(component_mask, M, (width, height))
                    
                    # Check overlap
                    new_overlap = cv2.bitwise_and(mask, shifted)
                    if np.sum(new_overlap > 0) == 0:
                        component_mask = shifted
                        break
        
        # Merge into original mask
        result = cv2.bitwise_or(mask, component_mask)
        
        return result
        
    def _apply_physical_constraints(self, mask, constraints):
        """Apply physical constraints (improved version, avoids holes)"""
        if mask is None:
            return None
            
        height, width = mask.shape
        original_mask = mask.copy()  # Save original mask for potential recovery
        
        # 1. Improved region constraint application
        if constraints.get("valid_regions") is not None:
            valid_region = constraints["valid_regions"]
            # Ensure dimensions match
            if valid_region.shape != mask.shape:
                valid_region = cv2.resize(valid_region, (width, height))
            
            # Improvement: fill holes in valid_region first for continuity
            valid_region_filled = self._fill_holes(valid_region)
            
            # Milder constraint: keep full mask if centroid is in valid region
            if np.sum(mask > 0) > 0:
                # Compute mask centroid
                y_indices, x_indices = np.where(mask > 0)
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                
                # If centroid in valid region, keep original mask
                if center_y < height and center_x < width and valid_region_filled[center_y, center_x] > 0:
                    # Centroid in valid region; keep mask with boundary check
                    overlap_ratio = np.sum(cv2.bitwise_and(mask, valid_region_filled) > 0) / np.sum(mask > 0)
                    if overlap_ratio > 0.3:  # At least 30% overlap
                        pass  # Keep original mask
                    else:
                        # Apply constraint and fill possible holes
                        mask = cv2.bitwise_and(mask, valid_region_filled)
                        mask = self._fill_holes(mask)
                else:
                    # Centroid outside valid region; move or constrain
                    mask = cv2.bitwise_and(mask, valid_region_filled)
                    mask = self._fill_holes(mask)
        
        # 2. Improved area constraints - more relaxed settings
        min_area, max_area = constraints.get("size_range", (10, width * height // 2))
        current_area = np.sum(mask > 0)
        
        # Use more relaxed minimum area threshold
        relaxed_min_area = max(20, min_area * 0.5)  # Lower minimum area requirement
        
        if current_area < relaxed_min_area and current_area > 0:
            # Area too small; expand with mild morphology
            target_scale = min(np.sqrt(relaxed_min_area / current_area), 3.0)  # Lower max expansion factor
            kernel_size = max(3, min(int(target_scale * 1.5), 5))  # Smaller kernel size
            # Ensure kernel_size is a valid positive integer
            kernel_size = max(3, min(kernel_size, 7))  # Limit to 3-7
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Use MORPH_CLOSE to fill small holes, then dilate
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, 
                                   np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        elif current_area > max_area and current_area > 0:
            # Area too large; shrink without direct erosion
            target_scale = max(np.sqrt(max_area / current_area), 0.7)  # Higher minimum shrink factor
            
            # Mild shrink: find largest connected component and resize
            mask = self._gentle_resize_mask(mask, target_scale)
        
        # 3. Final hole filling and cleanup
        mask = self._final_mask_cleanup(mask)
        
        # Ensure mask is binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _fill_holes(self, mask):
        """Fill holes in mask"""
        if np.sum(mask > 0) == 0:
            return mask
            
        # Use morphological closing to fill small holes
        kernel = np.ones((5, 5), np.uint8)
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # For larger holes, use flood fill
        h, w = mask.shape
        flood_filled = filled.copy()
        
        # Create mask 2px larger than image for flood fill
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Flood fill from corners (fill exterior region)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        
        # Invert to get interior holes
        holes = cv2.bitwise_not(flood_filled)
        
        # Fill holes into original mask
        result = cv2.bitwise_or(filled, holes)
        
        return result
    
    def _gentle_resize_mask(self, mask, scale_factor):
        """Gently resize mask while avoiding holes"""
        if np.sum(mask > 0) == 0:
            return mask
            
        # Find largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:  # Background only
            return mask
            
        # Find largest connected component (excluding background)
        largest_component_idx = 1
        largest_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_component_idx = i
        
        # Extract largest connected component
        largest_component = (labels == largest_component_idx).astype(np.uint8) * 255
        
        # Adjust based on scale_factor
        if scale_factor < 1.0:
            # Shrink: use opening
            kernel_size = max(3, int((1 / scale_factor - 1) * 2))
            # Ensure kernel_size is a valid positive integer
            kernel_size = max(3, min(kernel_size, 15))  # Limit to 3-15
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            result = cv2.morphologyEx(largest_component, cv2.MORPH_OPEN, kernel)
            
            # If result too small, retain more content
            if np.sum(result > 0) < largest_area * 0.3:
                kernel_size = max(3, kernel_size - 2)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                result = cv2.morphologyEx(largest_component, cv2.MORPH_OPEN, kernel)
        else:
            # Expand: use closing and dilation
            kernel_size = max(3, int((scale_factor - 1) * 2))
            # Ensure kernel_size is a valid positive integer
            kernel_size = max(3, min(kernel_size, 15))  # Limit to 3-15
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            result = cv2.morphologyEx(largest_component, cv2.MORPH_CLOSE, kernel)
            result = cv2.dilate(result, kernel, iterations=1)
        
        return result
    
    def _final_mask_cleanup(self, mask):
        """Final mask cleanup for quality"""
        if np.sum(mask > 0) == 0:
            print("mask_area 0")
            return mask
            
        # 1. Remove too-small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Use more reasonable minimum area threshold
        min_component_area = max(20, mask.shape[0] * mask.shape[1] * 0.0005)  # At least 20 pixels or 0.05% of total area
        
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
                cleaned_mask[labels == i] = 255
        
        # 2. Final hole filling
        if np.sum(cleaned_mask > 0) > 0:
            cleaned_mask = self._fill_holes(cleaned_mask)
        
        # 3. Light edge smoothing
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return cleaned_mask
    
    def _analyze_physical_constraints(self, sample_name, defect_name, mask_paths, defect_images):
        """Analyze physical constraints (improved, more reasonable ranges)"""
        # Initialize base constraints
        constraints = {
            "valid_regions": None,
            "size_range": (0, 0),
            "aspect_ratio_range": (0, 0),
            "complexity_range": (0, 0),
            "position_heatmap": None
        }
        
        # Analyze mask area statistics
        mask_areas = []
        mask_complexities = []
        mask_aspect_ratios = []
        position_maps = []
        
        width, height = 0, 0
        
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            # Record image dimensions
            if width == 0:
                height, width = mask.shape[:2]
                position_map = np.zeros((height, width), dtype=np.float32)
            else:
                mask = cv2.resize(mask, (width, height))
                
            # Compute mask area
            area = np.sum(mask > 0)
            mask_areas.append(area)
            
            # Compute aspect ratio
            if area > 0:
                y_indices, x_indices = np.where(mask > 0)
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                width_px = max_x - min_x + 1
                height_px = max_y - min_y + 1
                aspect_ratio = width_px / height_px if height_px > 0 else 1.0
                mask_aspect_ratios.append(aspect_ratio)
                
                # Update position heatmap
                position_map[y_indices, x_indices] += 1
                position_maps.append(position_map.copy())
                
            # Compute complexity
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours and area > 0:
                perimeter = 0
                for contour in contours:
                    perimeter += cv2.arcLength(contour, True)
                complexity = perimeter * perimeter / (4 * np.pi * area)
                mask_complexities.append(complexity)
        
        # Compute statistics (improved range calculation)
        if mask_areas:
            # Use conservative coefficients to avoid over-constraining
            min_area_base = max(int(np.percentile(mask_areas, 10)), 10)  # 10th percentile instead of 5th
            max_area_base = min(int(np.percentile(mask_areas, 90)), width * height * 0.5)  # 90th percentile
            
            constraints["size_range"] = (
                max(int(min_area_base * 0.5), 10),  # Allow smaller variation
                min(int(max_area_base * 2.0), width * height * 0.7)  # Allow larger variation
            )
        
        if mask_aspect_ratios:
            constraints["aspect_ratio_range"] = (
                max(float(np.percentile(mask_aspect_ratios, 5)), 0.1),
                min(float(np.percentile(mask_aspect_ratios, 95)), 10.0)
            )
        
        if mask_complexities:
            constraints["complexity_range"] = (
                max(float(np.percentile(mask_complexities, 5)), 1.0),
                min(float(np.percentile(mask_complexities, 95)), 10.0)
            )
        
        # Improved valid region generation
        if position_maps:
            avg_position_map = np.mean(np.array(position_maps), axis=0)
            if np.max(avg_position_map) > 0:
                avg_position_map = avg_position_map / np.max(avg_position_map)
            constraints["position_heatmap"] = avg_position_map
            
            # Generate more continuous valid region
            valid_region = (avg_position_map > 0.05).astype(np.uint8) * 255  # Lower threshold
            
            # Use larger kernel for continuity
            kernel = np.ones((25, 25), np.uint8)  # Larger kernel
            valid_region = cv2.morphologyEx(valid_region, cv2.MORPH_CLOSE, kernel, iterations=2)
            valid_region = cv2.dilate(valid_region, np.ones((15, 15), np.uint8), iterations=1)
            
            # Fill holes
            valid_region = self._fill_holes(valid_region)
            
            constraints["valid_regions"] = valid_region
        
        # If defect images provided, analyze content regions
        if defect_images and len(defect_images) > 0:
            self._analyze_content_regions_improved(defect_images, constraints, width, height)
        
        return constraints
    
    def _analyze_content_regions_improved(self, defect_images, constraints, width, height):
        """Improved content region analysis for continuous valid regions"""
        if not defect_images:
            return
            
        # Initialize content region mask
        content_mask = np.zeros((height, width), dtype=np.float32)
        
        for img_path in defect_images[:min(3, len(defect_images))]:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize image
            img = cv2.resize(img, (width, height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect content region using multiple methods
            # 1. Edge detection
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds to detect more edges
            
            # 2. Threshold segmentation (detect non-background regions)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. Combine edge and threshold results
            combined = cv2.bitwise_or(edges, thresh)
            
            # 4. Heavy dilation to create continuous region
            kernel = np.ones((15, 15), np.uint8)  # Larger kernel
            dilated = cv2.dilate(combined, kernel, iterations=3)
            
            # Accumulate into content mask
            content_mask += dilated
        
        # Normalize and post-process
        if np.max(content_mask) > 0:
            content_mask = content_mask / np.max(content_mask)
            
            # Lower threshold to include more area
            content_region = (content_mask > 0.1).astype(np.uint8) * 255
            
            # Large-scale morphology for continuity
            kernel = np.ones((30, 30), np.uint8)
            content_region = cv2.morphologyEx(content_region, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Fill all holes
            content_region = self._fill_holes(content_region)
            
            # Combine with existing valid region
            if constraints["valid_regions"] is not None:
                constraints["valid_regions"] = cv2.bitwise_or(
                    constraints["valid_regions"],
                    content_region
                )
            else:
                constraints["valid_regions"] = content_region


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LLM-guided adaptive mask generator')
    parser.add_argument('--data_root', required=True, help='Data root directory containing defect images')
    parser.add_argument('--mask_dir', default='mvtec_train_data', help='Original mask directory')
    parser.add_argument('--output_dir', default='llm_guided_masks', help='Output directory')
    parser.add_argument('--image_size', nargs=2, type=int, default=[256, 256], help='Output image size as width height')
    parser.add_argument('--num_simple', type=int, default=3, help='Number of simple masks to generate per category/defect')
    parser.add_argument('--num_complex', type=int, default=3, help='Number of complex masks to generate per category/defect')
    parser.add_argument('--selected_sample', type=str, default=None, help='Process only the specified sample type; process all if omitted')
    parser.add_argument('--selected_defect', type=str, default=None, help='Process only the specified defect type; process all if omitted')
    parser.add_argument('--api_key', type=str, default=None, help='No longer used')
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'], default='mvtec', help='Dataset type: mvtec or visa')
    parser.add_argument('--model_path', type=str, default=None, help='VLM model path or HuggingFace model ID (default in config.py)')
    
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
    # mis_info = {
    #     'AeBAD_S': ['ko'],
    #     'bracket_black': ['ko'],
    #     'bracket_white': ['ko'],
    #     'cable': ['ko'],
    #     'hazelnut': ['ko'],
    #     'leather': ['ko'],
    #     'tubes': ['ko'],
    #     'wood': ['ko'],
    # }
    
    # Select dataset info based on dataset type
    if args.dataset == 'mvtec':
        dataset_info = mvtec_info
        print("Using MVTec dataset configuration")
    elif args.dataset == 'visa':
        dataset_info = visa_info
        print("Using VISA dataset configuration")
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")
    
    # Save run parameters to JSON file
    params_file = os.path.join(args.output_dir, "generation_params.json")
    with open(params_file, 'w') as f:
        json.dump({
            "data_root": args.data_root,
            "mask_dir": args.mask_dir,
            "image_size": args.image_size,
            "num_simple": args.num_simple,
            "num_complex": args.num_complex,
            "selected_sample": args.selected_sample,
            "selected_defect": args.selected_defect,
            "dataset": args.dataset,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    # Initialize generator
    generator = LLMMaskGenerator(
        tuple(args.image_size),
        dataset_type=args.dataset,
        model_path=args.model_path,
    )
    
    # Statistics
    stats = {
        "total_simple": 0,
        "total_complex": 0,
        "total_categories": 0,
        "failed_categories": []
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
                # Generate mask
                simple_count, complex_count = generator.generate_masks(
                    sample_name,
                    defect_name,
                    args.data_root,
                    args.mask_dir,
                    args.output_dir,
                    args.num_simple,
                    args.num_complex
                )
                
                stats["total_simple"] += simple_count
                stats["total_complex"] += complex_count
                
                if simple_count + complex_count < args.num_simple + args.num_complex:
                    stats["failed_categories"].append(f"{sample_name}/{defect_name}")
                
                category_end_time = time.time()
                print(f"Finished {sample_name}/{defect_name}: generated {simple_count} simple and {complex_count} complex masks in {category_end_time - category_start_time:.2f}s")
                
            except Exception as e:
                print(f"Error processing {sample_name}/{defect_name}: {e}")
                stats["failed_categories"].append(f"{sample_name}/{defect_name}")
    
    total_end_time = time.time()
    
    # Save statistics
    stats["total_time"] = total_end_time - total_start_time
    stats_file = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n========== Generation complete ==========")
    print(f"Total generated: {stats['total_simple']} simple masks, {stats['total_complex']} complex masks")
    print(f"Processed {stats['total_categories']} categories, {len(stats['failed_categories'])} not fully successful")
    print(f"Total time: {stats['total_time']:.2f}s")
    print("===============================")

if __name__ == "__main__":
    main()

# MVTec dataset usage:
# python mask_generation.py \
#     --data_root mvtec_train_data \
#     --mask_dir mvtec_train_data \
#     --output_dir masks_mvtec \
#     --image_size 256 256 \
#     --num_simple 5 \
#     --num_complex 5 \
#     --dataset mvtec \
#     --selected_sample transistor \
#     --selected_defect misplaced

# VISA dataset usage:
# python mask_generation.py \
#     --data_root visa_train_data_balanced \
#     --mask_dir visa_train_data_balanced \
#     --output_dir masks_visa_xin \
#     --image_size 256 256 \
#     --num_simple 5 \
#     --num_complex 5 \
#     --dataset visa \
#     --selected_sample pcb1
# mis_info dataset usage:
# python mask_generation.py \
#     --data_root datasets/AD/mixdatasets_ref \
#     --mask_dir datasets/AD/mixdatasets_ref \
#     --output_dir masks_misdatasets \
#     --image_size 256 256 \
#     --num_simple 5 \
#     --num_complex 5 \
#     --dataset visa \
