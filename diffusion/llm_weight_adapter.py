import os
import glob

def parse_prompt_evaluation(prompt_file_path):
    """Parse a prompt evaluation file and extract sample, defect, and scores."""
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        sample_info = lines[0].split(': ')[1] if len(lines) > 0 and ': ' in lines[0] else None
        defect_info = lines[1].split(': ')[1] if len(lines) > 1 and ': ' in lines[1] else None

        eval_markers = ['===== Evaluation =====']
        eval_section = ""
        for marker in eval_markers:
            if marker in content:
                eval_section = content.split(marker)[1]
                break

        diversity_score = 0.0
        quality_scores = []
        avg_quality = 0.0
        comprehensive_score = 0.0

        label_map = {
            'diversity': ['Diversity score:'],
            'quality': ['Quality scores:'],
            'avg_quality': ['Average quality:'],
            'comprehensive': ['Comprehensive score:'],
        }

        for line in eval_section.strip().split('\n'):
            for key, labels in label_map.items():
                if any(label in line for label in labels):
                    value_part = line.split(':', 1)[1].strip()
                    if key == 'quality':
                        quality_scores = [float(s.strip()) for s in value_part.split(',')]
                    else:
                        score = float(value_part.split('/')[0].strip())
                        if key == 'diversity':
                            diversity_score = score
                        elif key == 'avg_quality':
                            avg_quality = score
                        elif key == 'comprehensive':
                            comprehensive_score = score

        return {
            'sample': sample_info,
            'defect': defect_info,
            'diversity_score': diversity_score,
            'quality_scores': quality_scores,
            'avg_quality': avg_quality,
            'comprehensive_score': comprehensive_score
        }
    except Exception as e:
        print(f"Failed to parse prompt evaluation file: {e}")
        return None

def calculate_adaptive_llm_weight(evaluation, base_weight=0.5, quality_impact=0.3, diversity_impact=0.2):
    """Compute adaptive llm_weight from prompt quality scores (range [0.1, 0.9])."""
    if not evaluation:
        return base_weight

    normalized_quality = evaluation['avg_quality'] / 10.0
    normalized_diversity = evaluation['diversity_score'] / 10.0
    normalized_comprehensive = evaluation['comprehensive_score'] / 10.0

    weight_adjustment = (
        normalized_comprehensive * 0.5 +
        normalized_quality * quality_impact +
        normalized_diversity * diversity_impact
    )

    adaptive_weight = base_weight + (weight_adjustment - 0.5) * 0.8
    return max(0.1, min(0.9, adaptive_weight))

def get_adaptive_llm_weight(sample_name, prompt_dir, default_weight=0.5, defect_name=None):
    """Find the prompt file for a sample/defect and return an adaptive llm_weight."""
    if defect_name:
        prompt_pattern = os.path.join(prompt_dir, f"{sample_name}_{defect_name}*_prompts.txt")
    else:
        prompt_pattern = os.path.join(prompt_dir, f"{sample_name}_prompts.txt")
    prompt_files = glob.glob(prompt_pattern)

    if not prompt_files:
        label = f"{sample_name}/{defect_name}" if defect_name else sample_name
        print(f"Warning: no prompt evaluation file for {label}, using default weight {default_weight}")
        return default_weight

    prompt_file = max(prompt_files, key=os.path.getmtime)
    print(f"Using prompt file: {prompt_file}")

    evaluation = parse_prompt_evaluation(prompt_file)
    if not evaluation:
        return default_weight

    adaptive_weight = calculate_adaptive_llm_weight(evaluation)
    print(f"Sample: {sample_name}, Defect: {defect_name}")
    print(f"Scores - comprehensive: {evaluation['comprehensive_score']}/10, "
          f"quality: {evaluation['avg_quality']}/10, diversity: {evaluation['diversity_score']}/10")
    print(f"Adaptive llm_weight: {adaptive_weight:.4f}")

    return adaptive_weight
