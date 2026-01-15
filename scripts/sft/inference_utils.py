"""
Utility functions and classes for Qwen3 inference and self-play evaluation.

This module contains:
- Configuration dataclasses
- Text extraction and parsing utilities
- Model generation helpers
- Filtering and similarity functions
- Prompt building utilities
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

# Import change parsing utilities (optional)
HAS_PARSE_CHANGES = False
try:
    from parse_changes import parse_raw_output, get_change_diff, format_change_log, compute_word_level_diff
    HAS_PARSE_CHANGES = True
except ImportError:
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from parse_changes import parse_raw_output, get_change_diff, format_change_log, compute_word_level_diff
        HAS_PARSE_CHANGES = True
    except ImportError:
        print(f"[WARNING] parse_changes.py not found (looked in: {script_dir}). Verbose diff logging disabled.")


# Constants
THINK_END_TOKEN_ID = 151668  # </think>
IM_END_TOKEN_ID = 151645  # <|im_end|>
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_GENERIC = "generic"
EARLY_STOPPING_TEXT_ASSESSOR = '\n</think>\n\n'
EARLY_STOPPING_TEXT_INJECTOR = '\n</think>\n\n'

DEFAULT_NOTE_FIELDS = [
    "correct_note",
    "note",
    "text",
    "original_note",
    "clinical_note",
]


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class GenerationConfig:
    """Configuration for model generation parameters."""
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    min_p: float = 0.05
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    do_sample: bool = True


@dataclass
class ThinkingConfig:
    """Configuration for Qwen thinking mode."""
    thinking_budget: int = 0
    assessor_thinking_budget: int = 256
    answer_tokens: int = 128
    early_stop_text_assessor: str = '\n</think>\n\n'
    early_stop_text_injector: str = '\n</think>\n\n'


@dataclass
class FilterConfig:
    """Configuration for note filtering in self-play."""
    min_jaccard: float = 0.85
    max_jaccard: float = 0.99
    max_word_edits: int = 6


@dataclass
class TokenConfig:
    """Special token IDs for Qwen."""
    think_end_token_id: int = 151668  # </think>
    im_end_token_id: int = 151645     # <|im_end|>


@dataclass
class FilterResult:
    """Result of similarity filtering."""
    passed: bool
    score_jaccard: Optional[float]
    reason: Optional[str]
    word_edits: int = 0
    sentences_changed: int = 0
    is_single_error: bool = False


# ============================================================================
# Prompt Building
# ============================================================================

def build_messages(
    mode: str,
    note: str,
    prompt_intent: str,
    assessor_prompts: Optional[Dict[str, str]] = None,
    injector_prompts: Optional[Dict[str, str]] = None,
    thinking_budget: int = 0,
    injector_is_correct: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """Build chat messages from JSON prompt configs. No hardcoded fallbacks."""
    if mode == "assessor":
        if not assessor_prompts:
            raise ValueError(
                "assessor_prompts is required. Provide --assessor-prompt-file pointing to a JSON "
                "file with 'system_prompt' and 'user_template' keys."
            )
        system_prompt = assessor_prompts["system_prompt"]
        user_content = assessor_prompts["user_template"].format(note=note)

    else:  # injector
        if not injector_prompts:
            raise ValueError(
                "injector_prompts is required. Provide --injector-prompt-file pointing to a JSON "
                "file with 'system_prompt_correct', 'system_prompt_incorrect', "
                "'injector_correct_template', 'injector_incorrect_template' keys."
            )
        if injector_is_correct is None:
            is_correct = "no clinical errors" in prompt_intent.lower()
        else:
            is_correct = injector_is_correct

        system_prompt = (
            injector_prompts["system_prompt_correct"]
            if is_correct
            else injector_prompts["system_prompt_incorrect"]
        )
        template = (
            injector_prompts["injector_correct_template"]
            if is_correct
            else injector_prompts["injector_incorrect_template"]
        )
        user_content = template.format(note=note, prompt_intent=prompt_intent)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_assessor_prompts(prompt_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if "system_prompt" not in prompts or "user_template" not in prompts:
        raise ValueError("Prompt file must include system_prompt and user_template.")
    return prompts


def load_injector_prompts(prompt_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    required = {
        "system_prompt_correct",
        "system_prompt_incorrect",
        "injector_correct_template",
        "injector_incorrect_template",
    }
    if not required.issubset(set(prompts.keys())):
        raise ValueError("Injector prompt file missing required keys.")
    return prompts


# ============================================================================
# Text Extraction and Parsing
# ============================================================================

def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer with multiple fallback patterns."""
    # Pattern registry: (regex_pattern, use_findall, result_mapper)
    patterns = [
        (r'final_answer:\s*"(CORRECT|INCORRECT)"', True, lambda m: m[-1].upper()),
        (r'final_answer:\s*(CORRECT|INCORRECT)', True, lambda m: m[-1].upper()),
        (r'Answer:\s*(CORRECT|INCORRECT)', True, lambda m: m[-1].upper()),
        (r'Assessment:\s*(CORRECT|INCORRECT)', True, lambda m: m[-1].upper()),
    ]

    for pattern, use_findall, mapper in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE) if use_findall else re.search(pattern, text, re.IGNORECASE)
        if matches:
            return mapper(matches)

    # Special case: Error Detected field (from model's old format)
    if re.search(r'Error Detected:\s*Yes', text, re.IGNORECASE):
        return "INCORRECT"
    elif re.search(r'Error Detected:\s*No', text, re.IGNORECASE):
        return "CORRECT"

    return None


def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_generated_note(note: str) -> str:
    if not note:
        return note
    note = strip_think_blocks(note)
    parts = re.split(r"\*\*Injected error\*\*:\s*", note, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    parts = re.split(r"Injected error:\s*", note, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    note = re.sub(r'final_answer:.*', '', note, flags=re.IGNORECASE | re.DOTALL).strip()
    return note.strip().strip('"').strip("'")


def _try_extract_pattern(pattern: str, text: str, group_idx: int = 1) -> Optional[str]:
    """Helper to try a regex pattern and return cleaned result."""
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(group_idx).strip()
        cleaned = clean_generated_note(candidate)
        return cleaned if cleaned else None
    return None


def extract_generated_note(text: str) -> Optional[str]:
    """Extract the generated_note section from injector output."""
    # Primary pattern: generated_note: ... final_answer:
    parts = re.split(r'generated_note:\s*', text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        after_label = parts[1]
        match = re.search(r'^(.*?)\s*final_answer:', after_label, re.DOTALL | re.IGNORECASE)
        generated = match.group(1).strip() if match else after_label.strip()
        if generated:
            cleaned = clean_generated_note(generated)
            if cleaned:
                return cleaned

    # Fallback patterns registry
    fallback_patterns = [
        r'Modified Note with Error:\s*"(.*?)"',
        r'Modified Note with Error:\s*(.*?)(?:\n\s*\d+\.\s+|Ground Truth|Error location|final_answer:|$)',
        r'Modified Note:\s*(.*?)(?:\n\s*\d+\.\s+|Ground Truth|Error location|final_answer:|$)',
    ]

    for pattern in fallback_patterns:
        result = _try_extract_pattern(pattern, text)
        if result:
            return result

    # Last resort: strip all markers and return remainder
    cleaned = strip_think_blocks(text)
    cleaned = re.sub(r'final_answer:.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = clean_generated_note(cleaned.strip())
    return cleaned if cleaned else None


# ============================================================================
# Model Detection and Qwen Utilities
# ============================================================================

def is_lora_adapter(path: Optional[str]) -> bool:
    if not path:
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def get_base_model_from_adapter(path: str) -> str:
    adapter_config_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("base_model_name_or_path", "")
    return ""


def detect_model_type(model_name: str, adapter_dir: Optional[str]) -> str:
    candidates = [model_name]
    if adapter_dir:
        candidates.append(adapter_dir)
        if is_lora_adapter(adapter_dir):
            base_model = get_base_model_from_adapter(adapter_dir)
            if base_model:
                candidates.append(base_model)

    for candidate in candidates:
        if "qwen" in candidate.lower():
            return MODEL_TYPE_QWEN
        config_path = os.path.join(candidate, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                model_type_str = config.get("model_type", "").lower()
                architectures = " ".join(config.get("architectures", [])).lower()
                if "qwen" in model_type_str or "qwen" in architectures:
                    return MODEL_TYPE_QWEN
            except Exception:
                pass

    return MODEL_TYPE_GENERIC


def normalize_qwen_thinking(thinking_content: str) -> str:
    if not thinking_content:
        return ""
    cleaned = re.sub(r"</?think>", "", thinking_content, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"Considering the limited time by the user, I have to give the solution based on the thinking directly now\.",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def strip_qwen_think_from_content(content: str) -> str:
    if not content:
        return content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^answer:\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


def parse_qwen3_output(tokenizer, input_ids, generated_ids) -> str:
    input_length = input_ids.shape[1]
    output_ids = generated_ids[0, input_length:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    thinking_content = normalize_qwen_thinking(thinking_content)
    content = strip_qwen_think_from_content(content)

    if thinking_content:
        return f"<think>{thinking_content}</think>\n{content}"
    return content


def parse_qwen3_output_with_length(tokenizer, input_length: int, generated_ids, original_seq_len: int = None) -> str:
    # If original_seq_len provided, use it to slice correctly with left-padding
    if original_seq_len is not None:
        output_ids = generated_ids[original_seq_len:].tolist()
    else:
        output_ids = generated_ids[input_length:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    thinking_content = normalize_qwen_thinking(thinking_content)
    content = strip_qwen_think_from_content(content)

    if thinking_content:
        return f"<think>{thinking_content}</think>\n{content}"
    return content


# ============================================================================
# Generation Utilities
# ============================================================================

def _build_generation_kwargs(
    gen_config: GenerationConfig,
    tokenizer,
    max_new_tokens: int = None,
    eos_token_id = None,
) -> Dict:
    """Build common generation kwargs from config."""
    return {
        "max_new_tokens": max_new_tokens or gen_config.max_new_tokens,
        "temperature": gen_config.temperature,
        "top_p": gen_config.top_p,
        "top_k": gen_config.top_k,
        "min_p": gen_config.min_p,
        "do_sample": gen_config.do_sample and gen_config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_id or tokenizer.eos_token_id,
        "repetition_penalty": gen_config.repetition_penalty,
    }


# ============================================================================
# Similarity and Filtering
# ============================================================================

def tokenize_for_jaccard(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(tokenize_for_jaccard(text1))
    set2 = set(tokenize_for_jaccard(text2))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def word_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in tokenize_for_jaccard(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def has_word_change(original: str, generated: str) -> bool:
    return word_counts(original) != word_counts(generated)


def passes_similarity_filter(
    original_note: str,
    generated_note: str,
    min_jaccard: float,
    max_jaccard: Optional[float] = None,
) -> FilterResult:
    """Legacy filter function for backward compatibility. Use apply_vcf instead."""
    if not generated_note:
        return FilterResult(passed=False, score_jaccard=None, reason="empty_generated")
    if not has_word_change(original_note, generated_note):
        return FilterResult(passed=False, score_jaccard=None, reason="no_word_change")
    score_jaccard = jaccard_similarity(original_note, generated_note)
    if score_jaccard < min_jaccard:
        return FilterResult(passed=False, score_jaccard=score_jaccard, reason="low_jaccard")
    if max_jaccard is not None and score_jaccard > max_jaccard:
        return FilterResult(passed=False, score_jaccard=score_jaccard, reason="too_similar")
    return FilterResult(passed=True, score_jaccard=score_jaccard, reason=None)


def apply_vcf(
    original_note: str,
    generated_note: str,
    min_jaccard: float = 0.85,
    max_jaccard: float = 0.99,
    max_word_edits: int = 6,
) -> FilterResult:
    """
    Apply Verifiable Curriculum Filter (VCF) to generated notes.

    Hard Filters (sequential - fail fast):
    1. Empty check
    2. Word change validation
    3. Jaccard similarity (0.85-0.99)
    4. Word edit count (≤6)
    5. Single error enforcement (single sentence changed)

    Args:
        original_note: Original clinical note
        generated_note: Generated/modified clinical note
        min_jaccard: Minimum Jaccard similarity threshold
        max_jaccard: Maximum Jaccard similarity threshold
        max_word_edits: Maximum number of word edits allowed

    Returns:
        FilterResult with pass/fail status and metadata
    """
    # Filter 1: Empty check
    if not generated_note:
        return FilterResult(
            passed=False,
            score_jaccard=None,
            reason="empty_generated",
        )

    # Filter 2: Word change validation
    if not has_word_change(original_note, generated_note):
        return FilterResult(
            passed=False,
            score_jaccard=None,
            reason="no_word_change",
        )

    # Filter 3: Jaccard similarity
    score_jaccard = jaccard_similarity(original_note, generated_note)
    if score_jaccard < min_jaccard:
        return FilterResult(
            passed=False,
            score_jaccard=score_jaccard,
            reason="low_jaccard",
        )
    if score_jaccard > max_jaccard:
        return FilterResult(
            passed=False,
            score_jaccard=score_jaccard,
            reason="too_similar",
        )

    # Filter 4 & 5: Word edit count + single error enforcement
    if HAS_PARSE_CHANGES:
        try:
            word_diff = compute_word_level_diff(original_note, generated_note)
            word_edits = word_diff["total_word_edits"]

            # Check edit count
            if word_edits > max_word_edits:
                return FilterResult(
                    passed=False,
                    score_jaccard=score_jaccard,
                    reason="too_many_edits",
                    word_edits=word_edits,
                    sentences_changed=0,
                )

            # Check single error constraint (single sentence changed)
            # Use a simple heuristic: count sentences with changes
            changes = word_diff.get("changes", [])
            if changes:
                # Count unique sentence-level changes (approximate)
                # If changes span > 1 distinct location, likely multiple sentences
                sentence_groups = []
                last_idx = -10
                for change in changes:
                    # Group changes that are close together (same sentence)
                    idx = change.get("idx", 0)
                    if idx - last_idx > 5:  # New sentence if gap > 5 words
                        sentence_groups.append([])
                    if sentence_groups:
                        sentence_groups[-1].append(change)
                    last_idx = idx

                sentences_changed = len(sentence_groups)

                # Allow edits in a single sentence region
                if sentences_changed > 1:
                    return FilterResult(
                        passed=False,
                        score_jaccard=score_jaccard,
                        reason="multiple_sentences_changed",
                        word_edits=word_edits,
                        sentences_changed=sentences_changed,
                    )

            return FilterResult(
                passed=True,
                score_jaccard=score_jaccard,
                reason=None,
                word_edits=word_edits,
                sentences_changed=1,
                is_single_error=True,
            )
        except Exception as e:
            # If parse_changes fails, fall back to basic filtering
            print(f"[WARNING] VCF word-level diff failed: {e}. Using basic filtering.")

    # Fallback if parse_changes not available
    return FilterResult(
        passed=True,
        score_jaccard=score_jaccard,
        reason=None,
    )


# ============================================================================
# Embedding Utilities
# ============================================================================

def load_embedding_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_texts(tokenizer, model, device: str, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    return float((emb_a * emb_b).sum().item())


# ============================================================================
# Data Loading and Writing
# ============================================================================

def select_note(record: Dict, note_field: Optional[str]) -> Optional[str]:
    if note_field:
        return record.get(note_field)
    for field in DEFAULT_NOTE_FIELDS:
        value = record.get(field)
        if value:
            return value
    return None


def load_records(jsonl_file: str) -> List[Dict]:
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def scenario_samples(records: List[Dict], scenario: str, num_samples: int) -> List[Dict]:
    if not records:
        return []
    if scenario == "assessor_correct":
        pool = [r for r in records if r.get("correct_note")]
    elif scenario == "assessor_incorrect":
        pool = [r for r in records if r.get("incorrect_note")]
    elif scenario == "injector_correct":
        pool = [r for r in records if r.get("correct_note")]
    else:
        pool = [r for r in records if r.get("correct_note") and r.get("incorrect_note")]
    if not pool:
        return []
    sample_count = min(num_samples, len(pool))
    import random
    return random.sample(pool, sample_count)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str, summary: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


# ============================================================================
# Result Summary
# ============================================================================

def summarize_results(rows: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for row in rows:
        scenario = row["scenario"]
        stats = summary.setdefault(
            scenario,
            {
                "total": 0,
                "correct": 0,
                "similarity_jaccard": [],
                "similarity_embedding": [],
            },
        )
        stats["total"] += 1
        stats["correct"] += int(bool(row.get("match")))
        if row.get("score_jaccard") is not None:
            stats["similarity_jaccard"].append(row["score_jaccard"])
        if row.get("score_embedding_cosine") is not None:
            stats["similarity_embedding"].append(row["score_embedding_cosine"])

    def agg_sim(values: List[float]) -> Dict:
        if not values:
            return {"avg": None, "min": None, "max": None}
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    assessor_correct = summary.get("assessor_correct", {"total": 0, "correct": 0})
    assessor_incorrect = summary.get("assessor_incorrect", {"total": 0, "correct": 0})
    injector_correct = summary.get("injector_correct", {"total": 0, "correct": 0})
    injector_incorrect = summary.get("injector_incorrect", {"total": 0, "correct": 0})

    total_assessor = assessor_correct["total"] + assessor_incorrect["total"]
    correct_assessor = assessor_correct["correct"] + assessor_incorrect["correct"]
    total_injector = injector_correct["total"] + injector_incorrect["total"]
    correct_injector = injector_correct["correct"] + injector_incorrect["correct"]

    return {
        "assessor": {
            "correct_notes": {
                "samples": assessor_correct["total"],
                "correct": assessor_correct["correct"],
                "accuracy": (assessor_correct["correct"] / assessor_correct["total"]) if assessor_correct["total"] else 0.0,
            },
            "incorrect_notes": {
                "samples": assessor_incorrect["total"],
                "correct": assessor_incorrect["correct"],
                "accuracy": (assessor_incorrect["correct"] / assessor_incorrect["total"]) if assessor_incorrect["total"] else 0.0,
            },
            "overall": {
                "samples": total_assessor,
                "correct": correct_assessor,
                "accuracy": (correct_assessor / total_assessor) if total_assessor else 0.0,
            },
        },
        "injector": {
            "generate_correct": {
                "samples": injector_correct["total"],
                "correct": injector_correct["correct"],
                "accuracy": (injector_correct["correct"] / injector_correct["total"]) if injector_correct["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_correct", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_correct", {}).get("similarity_embedding", [])),
            },
            "inject_errors": {
                "samples": injector_incorrect["total"],
                "correct": injector_incorrect["correct"],
                "accuracy": (injector_incorrect["correct"] / injector_incorrect["total"]) if injector_incorrect["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_incorrect", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_incorrect", {}).get("similarity_embedding", [])),
            },
            "overall": {
                "samples": total_injector,
                "correct": correct_injector,
                "accuracy": (correct_injector / total_injector) if total_injector else 0.0,
                "similarity_jaccard": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_jaccard", []) +
                     summary.get("injector_incorrect", {}).get("similarity_jaccard", []))
                ),
                "similarity_embedding": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_embedding", []) +
                     summary.get("injector_incorrect", {}).get("similarity_embedding", []))
                ),
            },
        },
    }
