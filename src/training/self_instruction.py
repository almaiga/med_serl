"""
Self-Instruction Module for MedSeRL.

Aligned with SeRL paper: "Self-play Reinforcement Learning for LLMs with Limited Data"

Key components:
1. Few-shot prompt building from MEDEC seed data
2. Rouge-L diversity filtering (threshold: 0.7)
3. Difficulty filtering (accuracy bounds: 0.2-0.8)
4. Dynamic accumulation until batch is full
5. Sample expiration after N steps

This generates NEW clinical notes for the model to analyze during RL,
expanding the training distribution beyond the original MEDEC dataset.
"""

import random
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from rouge_score import rouge_scorer


# Error types (for reference/logging, not stratification)
ERROR_TYPES = [
    "Diagnosis",
    "Pharmacotherapy", 
    "Management",
    "Treatment",
    "CausalOrganism"
]

# Default configuration aligned with SeRL paper
DEFAULT_CONFIG = {
    "rouge_threshold": 0.7,           # Diversity: reject if similarity > 0.7
    "difficulty_lower": 0.2,          # Too hard if accuracy < 0.2
    "difficulty_upper": 0.8,          # Too easy if accuracy > 0.8
    "num_few_shot": 4,                # Number of few-shot examples
    "num_from_generated": 2,          # How many from generated pool
    "expiration_steps": 2,            # Remove samples after N steps
    "n_samples_for_difficulty": 4,    # Samples for difficulty estimation
}


@dataclass
class GeneratedSample:
    """A generated clinical note with metadata."""
    clinical_note: str
    has_error: bool
    error_type: Optional[str]
    source: str = "self_instruction"
    created_at_step: int = 0
    difficulty_score: float = 0.5  # Estimated accuracy


@dataclass 
class SelfInstructionState:
    """Maintains state across training for self-instruction."""
    generated_pool: List[GeneratedSample] = field(default_factory=list)
    seed_notes: List[str] = field(default_factory=list)  # For Rouge-L comparison
    scorer: Any = None  # Rouge scorer instance
    
    def __post_init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def initialize_self_instruction(
    seed_error_samples: List[Dict],
    seed_correct_samples: List[Dict],
    config: Optional[Dict] = None
) -> Tuple[SelfInstructionState, Dict]:
    """
    Initialize self-instruction with MEDEC seed data.
    
    Args:
        seed_error_samples: MEDEC samples with errors
        seed_correct_samples: MEDEC samples without errors
        config: Optional config overrides
        
    Returns:
        Tuple of (state, config)
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    
    state = SelfInstructionState()
    
    # Store seed notes for Rouge-L comparison
    for sample in seed_error_samples:
        note = sample.get("original_text", sample.get("text", ""))
        if note:
            state.seed_notes.append(note)
    
    for sample in seed_correct_samples:
        note = sample.get("original_text", sample.get("text", ""))
        if note:
            state.seed_notes.append(note)
    
    print(f"Self-instruction initialized with {len(state.seed_notes)} seed notes")
    print(f"Config: rouge_threshold={cfg['rouge_threshold']}, "
          f"difficulty_bounds=[{cfg['difficulty_lower']}, {cfg['difficulty_upper']}]")
    
    return state, cfg


def build_few_shot_prompt(
    seed_error_samples: List[Dict],
    seed_correct_samples: List[Dict],
    state: SelfInstructionState,
    config: Dict
) -> Tuple[str, List[Dict]]:
    """
    Build a few-shot prompt for generating new clinical notes.
    
    Samples from:
    - Previously generated pool (if available)
    - MEDEC seed data (balanced error/correct)
    
    Args:
        seed_error_samples: MEDEC samples with errors
        seed_correct_samples: MEDEC samples without errors
        state: Self-instruction state
        config: Configuration dict
        
    Returns:
        Tuple of (prompt_string, selected_examples)
    """
    num_few_shot = config["num_few_shot"]
    num_from_generated = config["num_from_generated"]
    
    selected_examples = []
    
    # 1. Sample from generated pool (if available)
    if len(state.generated_pool) >= num_from_generated:
        generated_samples = random.sample(state.generated_pool, num_from_generated)
        for gs in generated_samples:
            selected_examples.append({
                "text": gs.clinical_note,
                "has_error": gs.has_error,
                "error_type": gs.error_type,
                "source": "generated"
            })
    
    # 2. Fill remaining slots from MEDEC seed (balanced)
    remaining = num_few_shot - len(selected_examples)
    if remaining > 0:
        num_error = remaining // 2
        num_correct = remaining - num_error
        
        # Sample error examples
        if seed_error_samples and num_error > 0:
            error_samples = random.sample(
                seed_error_samples, 
                min(num_error, len(seed_error_samples))
            )
            for s in error_samples:
                selected_examples.append({
                    "text": s.get("original_text", s.get("text", "")),
                    "has_error": True,
                    "error_type": s.get("meta", {}).get("error_type", s.get("error_type")),
                    "source": "seed"
                })
        
        # Sample correct examples
        if seed_correct_samples and num_correct > 0:
            correct_samples = random.sample(
                seed_correct_samples,
                min(num_correct, len(seed_correct_samples))
            )
            for s in correct_samples:
                selected_examples.append({
                    "text": s.get("original_text", s.get("text", "")),
                    "has_error": False,
                    "error_type": None,
                    "source": "seed"
                })
    
    # Shuffle examples
    random.shuffle(selected_examples)
    
    # Build prompt
    prompt = "Here are examples of clinical notes. Some contain medical errors, some are correct:\n\n"
    
    for i, ex in enumerate(selected_examples, 1):
        label = "Contains error" if ex["has_error"] else "Correct"
        if ex["has_error"] and ex.get("error_type"):
            label += f" ({ex['error_type']})"
        
        # Truncate long notes for the prompt
        note_text = ex["text"][:500] + "..." if len(ex["text"]) > 500 else ex["text"]
        prompt += f"Example {i} [{label}]:\n{note_text}\n\n"
    
    prompt += """Generate a new clinical note that is different from the examples above.
The note should be realistic and either:
- Contain a subtle but clinically significant medical error, OR
- Be completely correct with no errors

Write only the clinical note (3-5 sentences), no explanations or labels:"""
    
    return prompt, selected_examples


def compute_rouge_similarity(
    new_note: str,
    state: SelfInstructionState
) -> float:
    """
    Compute maximum Rouge-L similarity against all existing notes.
    
    Args:
        new_note: The newly generated note
        state: Self-instruction state with seed notes and generated pool
        
    Returns:
        Maximum Rouge-L F-measure (0.0 to 1.0)
    """
    max_similarity = 0.0
    
    # Check against seed notes
    for seed_note in state.seed_notes:
        score = state.scorer.score(new_note, seed_note)
        similarity = score["rougeL"].fmeasure
        max_similarity = max(max_similarity, similarity)
        
        # Early exit if already above threshold
        if max_similarity > 0.9:
            return max_similarity
    
    # Check against generated pool
    for generated in state.generated_pool:
        score = state.scorer.score(new_note, generated.clinical_note)
        similarity = score["rougeL"].fmeasure
        max_similarity = max(max_similarity, similarity)
        
        if max_similarity > 0.9:
            return max_similarity
    
    return max_similarity


def passes_rouge_filter(
    new_note: str,
    state: SelfInstructionState,
    config: Dict
) -> Tuple[bool, float]:
    """
    Check if note passes Rouge-L diversity filter.
    
    Args:
        new_note: The newly generated note
        state: Self-instruction state
        config: Configuration dict
        
    Returns:
        Tuple of (passes_filter, max_similarity)
    """
    max_sim = compute_rouge_similarity(new_note, state)
    passes = max_sim < config["rouge_threshold"]
    return passes, max_sim


def estimate_difficulty(
    model,
    tokenizer,
    clinical_note: str,
    config: Dict,
    device: str = "cuda"
) -> Tuple[float, bool, str]:
    """
    Estimate difficulty by sampling model responses.
    
    Difficulty = accuracy of model on this note.
    - Too easy (> upper bound): model always gets it right
    - Too hard (< lower bound): model always gets it wrong
    - Good difficulty: model sometimes right, sometimes wrong
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        clinical_note: The note to evaluate
        config: Configuration dict
        device: Device to run on
        
    Returns:
        Tuple of (accuracy, has_error_prediction, sample_response)
    """
    import torch
    
    n_samples = config["n_samples_for_difficulty"]
    
    # Build evaluation prompt
    prompt = f"""You are a healthcare professional analyzing medical notes.

Clinical Note:
{clinical_note}

Does this clinical note contain a medical error? 
Provide brief reasoning in <think> tags, then your answer in <answer> tags.
Answer with <answer>CORRECT</answer> if no error, or <answer>INCORRECT</answer> if there is an error:"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    
    predictions = []
    sample_response = ""
    
    with torch.no_grad():
        for i in range(n_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            if i == 0:
                sample_response = response
            
            # Parse prediction
            response_lower = response.lower()
            if "<answer>incorrect</answer>" in response_lower:
                predictions.append("INCORRECT")
            elif "<answer>correct</answer>" in response_lower:
                predictions.append("CORRECT")
            elif "incorrect" in response_lower:
                predictions.append("INCORRECT")
            else:
                predictions.append("CORRECT")
    
    # Majority vote to determine "ground truth" for this generated note
    incorrect_count = predictions.count("INCORRECT")
    correct_count = predictions.count("CORRECT")
    
    # The majority prediction becomes the "label"
    has_error = incorrect_count > correct_count
    
    # Accuracy = how consistent the model is (proxy for difficulty)
    # High consistency = easy, low consistency = good difficulty
    majority_count = max(incorrect_count, correct_count)
    accuracy = majority_count / n_samples
    
    return accuracy, has_error, sample_response


def passes_difficulty_filter(
    accuracy: float,
    config: Dict
) -> bool:
    """
    Check if sample passes difficulty filter.
    
    Args:
        accuracy: Model accuracy on this sample
        config: Configuration dict
        
    Returns:
        True if within difficulty bounds
    """
    return config["difficulty_lower"] <= accuracy <= config["difficulty_upper"]


def expire_old_samples(
    state: SelfInstructionState,
    current_step: int,
    config: Dict
) -> int:
    """
    Remove samples that have expired (older than N steps).
    
    Args:
        state: Self-instruction state
        current_step: Current training step
        config: Configuration dict
        
    Returns:
        Number of samples removed
    """
    expiration_steps = config["expiration_steps"]
    
    original_count = len(state.generated_pool)
    state.generated_pool = [
        s for s in state.generated_pool
        if current_step - s.created_at_step <= expiration_steps
    ]
    
    removed = original_count - len(state.generated_pool)
    return removed


def generate_and_filter_batch(
    model,
    tokenizer,
    seed_error_samples: List[Dict],
    seed_correct_samples: List[Dict],
    state: SelfInstructionState,
    config: Dict,
    target_batch_size: int,
    current_step: int,
    device: str = "cuda",
    max_attempts: int = 50,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate samples until we have target_batch_size that pass all filters.
    
    This is the main entry point for self-instruction during RL training.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        seed_error_samples: MEDEC samples with errors
        seed_correct_samples: MEDEC samples without errors
        state: Self-instruction state
        config: Configuration dict
        target_batch_size: How many samples we need
        current_step: Current training step (for expiration)
        device: Device to run on
        max_attempts: Maximum generation attempts
        verbose: Print progress
        
    Returns:
        List of sample dicts ready for training
    """
    import torch
    
    # First, expire old samples
    expired = expire_old_samples(state, current_step, config)
    if expired > 0 and verbose:
        print(f"  Expired {expired} old samples from generated pool")
    
    accepted_samples = []
    stats = {
        "attempts": 0,
        "rouge_rejected": 0,
        "difficulty_rejected_easy": 0,
        "difficulty_rejected_hard": 0,
        "length_rejected": 0,
        "accepted": 0
    }
    
    while len(accepted_samples) < target_batch_size and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        
        # Build few-shot prompt
        prompt, _ = build_few_shot_prompt(
            seed_error_samples,
            seed_correct_samples,
            state,
            config
        )
        
        # Generate new clinical note
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_note = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Clean up - remove any trailing explanations
        if "\n\n" in generated_note:
            generated_note = generated_note.split("\n\n")[0].strip()
        
        # Length filter
        if len(generated_note) < 50:
            stats["length_rejected"] += 1
            continue
        
        # Rouge-L diversity filter
        passes_rouge, similarity = passes_rouge_filter(generated_note, state, config)
        if not passes_rouge:
            stats["rouge_rejected"] += 1
            continue
        
        # Difficulty filter
        accuracy, has_error, sample_response = estimate_difficulty(
            model, tokenizer, generated_note, config, device
        )
        
        if accuracy > config["difficulty_upper"]:
            stats["difficulty_rejected_easy"] += 1
            continue
        elif accuracy < config["difficulty_lower"]:
            stats["difficulty_rejected_hard"] += 1
            continue
        
        # Passed all filters!
        stats["accepted"] += 1
        
        # Determine error type from response if has_error
        error_type = None
        if has_error:
            # Try to infer error type from the note content
            # This is approximate - the model's majority vote determines has_error
            error_type = "Unknown"  # Could be enhanced with classification
        
        # Create sample
        generated_sample = GeneratedSample(
            clinical_note=generated_note,
            has_error=has_error,
            error_type=error_type,
            source="self_instruction",
            created_at_step=current_step,
            difficulty_score=accuracy
        )
        
        # Add to generated pool for future few-shot sampling
        state.generated_pool.append(generated_sample)
        
        # Add to batch
        accepted_samples.append({
            "original_text": generated_note,
            "text": generated_note,
            "meta": {
                "has_error": has_error,
                "error_type": error_type,
                "source": "self_instruction"
            },
            "ground_truth": {
                "has_error": has_error,
                "error_type": error_type,
                "source": "self_instruction"
            },
            "quadrant": "self_instruction",
            "difficulty": accuracy,
            "rouge_similarity": similarity
        })
        
        # Clear cache
        torch.cuda.empty_cache()
    
    if verbose:
        print(f"  Self-instruction: {stats['accepted']}/{stats['attempts']} accepted "
              f"(rouge_rej={stats['rouge_rejected']}, "
              f"easy_rej={stats['difficulty_rejected_easy']}, "
              f"hard_rej={stats['difficulty_rejected_hard']}, "
              f"len_rej={stats['length_rejected']})")
    
    return accepted_samples


def get_self_instruction_stats(state: SelfInstructionState) -> Dict:
    """
    Get statistics about the self-instruction state.
    
    Args:
        state: Self-instruction state
        
    Returns:
        Dict with statistics
    """
    if not state.generated_pool:
        return {
            "pool_size": 0,
            "error_count": 0,
            "correct_count": 0,
            "avg_difficulty": 0.0
        }
    
    error_count = sum(1 for s in state.generated_pool if s.has_error)
    correct_count = len(state.generated_pool) - error_count
    avg_difficulty = sum(s.difficulty_score for s in state.generated_pool) / len(state.generated_pool)
    
    return {
        "pool_size": len(state.generated_pool),
        "error_count": error_count,
        "correct_count": correct_count,
        "avg_difficulty": avg_difficulty,
        "seed_notes_count": len(state.seed_notes)
    }
