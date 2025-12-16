#!/usr/bin/env python3
"""
Medical Error Detection Inference Script

Tests different model versions on MEDEC test data:
- Qwen3 models (Qwen/Qwen3-4B, etc.)
- MedGemma models (google/medgemma-4b-it, google/medgemma-4b-pt)
- Fine-tuned models (from SFT/GRPO)

Uses CoT prompting with few-shot examples for error detection.
Supports both Qwen3 thinking format and standard generation.
"""

import os
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Qwen3 special token IDs (from official documentation)
THINK_END_TOKEN_ID = 151668  # </think>
IM_END_TOKEN_ID = 151645  # <|im_end|>

# Model type detection
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_MEDGEMMA = "medgemma"  # Original multimodal MedGemma from HF
MODEL_TYPE_GEMMA = "gemma"  # Fine-tuned Gemma/MedGemma (text-only CausalLM)
MODEL_TYPE_GENERIC = "generic"


def is_lora_adapter(model_path: str) -> bool:
    """Check if the path contains a LoRA adapter."""
    adapter_config = os.path.join(model_path, "adapter_config.json")
    return os.path.exists(adapter_config)


def get_base_model_from_adapter(model_path: str) -> str:
    """Get the base model name from adapter_config.json."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "")
    return ""


def detect_model_type(model_path: str) -> str:
    """Detect model type from path/name or model config."""
    model_path_lower = model_path.lower()
    
    # Check if it's a LoRA adapter and get base model
    if is_lora_adapter(model_path):
        base_model = get_base_model_from_adapter(model_path)
        if base_model:
            model_path_lower = base_model.lower()
    
    if "medgemma" in model_path_lower:
        return MODEL_TYPE_GEMMA  # Fine-tuned MedGemma uses CausalLM
    elif "gemma" in model_path_lower:
        return MODEL_TYPE_GEMMA
    elif "qwen" in model_path_lower:
        return MODEL_TYPE_QWEN
    else:
        # Check config.json for model type if it's a local path
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_type_str = config.get("model_type", "").lower()
                architectures = config.get("architectures", [])
                arch_str = " ".join(architectures).lower()
                
                if "qwen" in model_type_str or "qwen" in arch_str:
                    return MODEL_TYPE_QWEN
                elif "gemma" in model_type_str or "gemma" in arch_str:
                    return MODEL_TYPE_GEMMA
            except Exception:
                pass
        return MODEL_TYPE_GENERIC


def load_model_and_tokenizer(model_path: str, model_type: str):
    """Load model and tokenizer from local or HF path based on model type."""
    print(f"Loading model from: {model_path}")
    print(f"Model type: {model_type}")
    
    # Check if this is a LoRA adapter
    is_adapter = is_lora_adapter(model_path)
    if is_adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required to load LoRA adapters. Install with: pip install peft")
        base_model_path = get_base_model_from_adapter(model_path)
        print(f"🔧 Detected LoRA adapter, base model: {base_model_path}")
    
    if model_type == MODEL_TYPE_MEDGEMMA and not is_adapter:
        # Original MedGemma uses AutoModelForImageTextToText and AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✅ MedGemma multimodal model loaded successfully")
        return model, processor
    else:
        # Load base model first if it's a LoRA adapter
        if is_adapter:
            base_model_path = get_base_model_from_adapter(model_path)
            print(f"📦 Loading base model: {base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"🔗 Loading LoRA adapter from: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            # Merge adapter for faster inference
            print(f"⚡ Merging adapter for faster inference...")
            model = model.merge_and_unload()
            
            # Load tokenizer from adapter path (has chat template) or base model
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        else:
            # Standard model loading
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_type_name = "Gemma" if model_type == MODEL_TYPE_GEMMA else model_type.capitalize()
        adapter_info = " (with LoRA)" if is_adapter else ""
        print(f"✅ {model_type_name} model loaded successfully{adapter_info}")
        return model, tokenizer


def load_test_data(dataset_name: str = "all") -> pd.DataFrame:
    """
    Load MEDEC test data.
    
    Args:
        dataset_name: "ms", "uw", or "all"
    """
    dfs = []
    
    # Try multiple possible data paths
    data_dirs = ["data_raw/MEDEC", "data_copy/MEDEC", "data/MEDEC"]
    
    if dataset_name in ["ms", "all"]:
        ms_loaded = False
        for data_dir in data_dirs:
            ms_path = f"{data_dir}/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv"
            if os.path.exists(ms_path):
                df_ms = pd.read_csv(ms_path)
                # Filter out rows with null Text ID
                df_ms = df_ms[df_ms['Text ID'].notna()].copy()
                df_ms['dataset'] = 'MS'
                dfs.append(df_ms)
                print(f"✅ Loaded MS test set: {len(df_ms)} examples from {ms_path}")
                ms_loaded = True
                break
        if not ms_loaded and dataset_name == "ms":
            print(f"⚠️ MS test set not found in any of: {data_dirs}")
    
    if dataset_name in ["uw", "all"]:
        uw_loaded = False
        for data_dir in data_dirs:
            uw_path = f"{data_dir}/MEDEC-UW/MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv"
            if os.path.exists(uw_path):
                df_uw = pd.read_csv(uw_path)
                # Filter out rows with null Text ID
                df_uw = df_uw[df_uw['Text ID'].notna()].copy()
                df_uw['dataset'] = 'UW'
                dfs.append(df_uw)
                print(f"✅ Loaded UW test set: {len(df_uw)} examples from {uw_path}")
                uw_loaded = True
                break
        if not uw_loaded and dataset_name == "uw":
            print(f"⚠️ UW test set not found in any of: {data_dirs}")
    
    if not dfs:
        raise FileNotFoundError(f"No test data found for dataset: {dataset_name}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total test examples: {len(df)}")
    
    return df


def get_few_shot_examples() -> List[Dict[str, str]]:
    """
    Return few-shot examples for error detection.
    Uses real examples from MEDEC-MS training set.
    """
    return [
        {
            "note": "A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum. He works as a commercial fisherman on Lake Superior. Current medications include metoprolol and warfarin. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field. After reviewing imaging, the causal pathogen was determined to be Haemophilus influenzae. An x-ray of the chest showed consolidation of the right upper lobe.",
            "reasoning": "Analyzing this pneumonia case:\n1. Clinical presentation: fever, productive cough, vital signs showing tachypnea and hypotension\n2. Physical exam: increased fremitus and bronchial breath sounds in right middle lung field\n3. Imaging: consolidation of right UPPER lobe\n4. Stated pathogen: Haemophilus influenzae\n\nDiscrepancy detected: Physical exam findings are in the right MIDDLE lung field, but X-ray shows RIGHT UPPER lobe consolidation. This anatomical mismatch is concerning. Additionally, for a lobar pneumonia with these classic findings, Streptococcus pneumoniae is more likely than H. influenzae in an adult.",
            "label": "INCORRECT",
            "explanation": "Anatomical discrepancy between physical exam (right middle field) and imaging (right upper lobe), plus incorrect pathogen identification"
        },
        {
            "note": "A 53-year-old man comes to the physician because of a 1-day history of fever and chills, severe malaise, and cough with yellow-green sputum. He works as a commercial fisherman on Lake Superior. Current medications include metoprolol and warfarin. His temperature is 38.5 C (101.3 F), pulse is 96/min, respirations are 26/min, and blood pressure is 98/62 mm Hg. Examination shows increased fremitus and bronchial breath sounds over the right middle lung field. An x-ray of the chest shows consolidation of the right upper lobe. The causal pathogen is Streptococcus pneumoniae.",
            "reasoning": "Reviewing this pneumonia case:\n1. Clinical presentation: acute onset fever, chills, productive cough with purulent sputum\n2. Vital signs: fever 38.5C, tachypnea (26/min), relative hypotension\n3. Physical exam: increased fremitus and bronchial breath sounds - classic for consolidation\n4. Imaging: right upper lobe consolidation\n5. Pathogen: Streptococcus pneumoniae\n\nAll findings are consistent with community-acquired pneumonia. S. pneumoniae is the most common cause of lobar pneumonia in adults. Physical exam and imaging findings align appropriately. No errors detected.",
            "label": "CORRECT",
            "explanation": "Consistent clinical presentation of community-acquired pneumonia with appropriate pathogen identification"
        },
        {
            "note": "A 9-year-old girl is brought to the pediatrician by her mother who reports that the girl has been complaining of genital itching over the past few days. She states she has noticed her daughter scratching her buttocks and anus for the past week; however, now she is scratching her groin quite profusely as well. The mother notices that symptoms seem to be worse at night. The girl is otherwise healthy, is up to date on her vaccinations, and feels well. She was recently treated with amoxicillin for a middle ear infection. The child also had a recent bout of diarrhea that was profuse and watery that seems to be improving. Her temperature is 98.5 F (36.9 C), pulse is 88/min, respirations are 18/min, and blood pressure is 110/72 mm Hg. Suspected of infection with Giardia lamblia.",
            "reasoning": "Analyzing this pediatric case:\n1. Chief complaint: perianal and genital itching, worse at night\n2. Key symptom: nocturnal pruritus (worse at night) - classic sign\n3. Location: anus, buttocks, groin\n4. Recent history: recent diarrhea (improving), recent antibiotics\n5. Suspected diagnosis: Giardia lamblia\n\nError identified: The clinical presentation strongly suggests pinworm (Enterobius vermicularis) infection, NOT Giardia. Nocturnal perianal itching is pathognomonic for pinworms. Giardia causes GI symptoms (diarrhea, bloating) but not the characteristic nighttime perianal itching. This is a diagnostic error.",
            "label": "INCORRECT",
            "explanation": "Incorrect diagnosis - symptoms indicate Enterobius vermicularis (pinworm) infection, not Giardia lamblia"
        },
        {
            "note": "A 9-year-old girl is brought to the pediatrician by her mother who reports that the girl has been complaining of genital itching over the past few days. She states she has noticed her daughter scratching her buttocks and anus for the past week; however, now she is scratching her groin quite profusely as well. The mother notices that symptoms seem to be worse at night. The girl is otherwise healthy, is up to date on her vaccinations, and feels well. She was recently treated with amoxicillin for a middle ear infection. The child also had a recent bout of diarrhea that was profuse and watery that seems to be improving. Her temperature is 98.5 F (36.9 C), pulse is 88/min, respirations are 18/min, and blood pressure is 110/72 mm Hg. Suspected of infection with Enterobius vermicularis.",
            "reasoning": "Reviewing this pediatric case:\n1. Presenting complaint: perianal and genital itching for one week\n2. Key feature: symptoms worse at night (nocturnal pruritus)\n3. Location: anus, buttocks, groin area\n4. Patient: otherwise healthy 9-year-old girl\n5. Diagnosis: Enterobius vermicularis (pinworm)\n\nThis is correct. Nocturnal perianal itching is the hallmark symptom of pinworm infection. Female pinworms migrate to the perianal area at night to lay eggs, causing intense itching. The diagnosis matches the clinical presentation perfectly. No errors detected.",
            "label": "CORRECT",
            "explanation": "Correct diagnosis of Enterobius vermicularis based on classic nocturnal perianal itching"
        }
    ]


def build_error_detection_prompt(
    note: str,
    use_few_shot: bool = True,
    use_cot: bool = True,
    model_type: str = MODEL_TYPE_QWEN
) -> Tuple[List[Dict[str, str]], bool]:
    """
    Build prompt for error detection with optional few-shot and CoT.
    
    Returns: (messages list for chat template, enable_thinking flag)
    """
    system_prompt = """You are an expert medical error detection system. Your task is to carefully analyze medical notes for potential clinical errors, diagnostic mistakes, or treatment inaccuracies.

Classify each note as:
- "CORRECT": No medical errors detected
- "INCORRECT": Contains medical errors, diagnostic mistakes, or treatment inaccuracies

Always explain your reasoning."""

    # MedGemma uses different message format with content as list
    if model_type == MODEL_TYPE_MEDGEMMA:
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    else:
        messages = [{"role": "system", "content": system_prompt}]
    
    # Add few-shot examples
    if use_few_shot:
        examples = get_few_shot_examples()
        for ex in examples:
            user_msg = f"Analyze this medical note:\n\n{ex['note']}"
            
            if use_cot and model_type == MODEL_TYPE_QWEN:
                # Use Qwen3's native thinking format for few-shot examples
                assistant_msg = f"<think>\n{ex['reasoning']}\n</think>\n\nLabel: {ex['label']}\nExplanation: {ex['explanation']}"
            elif use_cot:
                # For MedGemma and others, include reasoning inline
                assistant_msg = f"Reasoning: {ex['reasoning']}\n\nLabel: {ex['label']}\nExplanation: {ex['explanation']}"
            else:
                assistant_msg = f"Label: {ex['label']}\nExplanation: {ex['explanation']}"
            
            if model_type == MODEL_TYPE_MEDGEMMA:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_msg}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]})
            else:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the actual query
    query = f"Analyze this medical note:\n\n{note}"
    if model_type == MODEL_TYPE_MEDGEMMA:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
    else:
        messages.append({"role": "user", "content": query})
    
    # Return enable_thinking flag for Qwen3's native CoT (only for Qwen models)
    enable_thinking = use_cot and model_type == MODEL_TYPE_QWEN
    return messages, enable_thinking


def parse_qwen3_output(tokenizer, input_ids, generated_ids) -> Tuple[str, str]:
    """
    Parse Qwen3 output using official method (token-based parsing).
    
    Returns: (thinking_content, content)
    """
    input_length = input_ids.shape[1]
    output_ids = generated_ids[0, input_length:].tolist()
    
    # Parse thinking content using token ID (official Qwen3 method)
    try:
        # Find </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        # No thinking content found
        index = 0
    
    thinking_content = tokenizer.decode(
        output_ids[:index], 
        skip_special_tokens=True
    ).strip("\n")
    
    content = tokenizer.decode(
        output_ids[index:], 
        skip_special_tokens=True
    ).strip("\n")
    
    return thinking_content, content


def parse_response(thinking: str, content: str) -> Tuple[str, str, str]:
    """
    Parse model response to extract thinking, label, and explanation.
    
    Args:
        thinking: The thinking content (from <think> block)
        content: The final response content
    
    Returns: (thinking, label, explanation)
    """
    label = "Unknown"
    explanation = ""
    
    # Extract label and explanation from the content
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Look for label
        if 'label:' in line_lower or 'answer:' in line_lower:
            label_text = line.split(':', 1)[1].strip()
            # Normalize label
            if 'correct' in label_text.lower() and 'incorrect' not in label_text.lower():
                label = 'CORRECT'
            elif 'incorrect' in label_text.lower():
                label = 'INCORRECT'
        
        # Look for explanation
        if 'explanation:' in line_lower:
            explanation = line.split(':', 1)[1].strip()
            # Get rest of explanation if multi-line
            if i + 1 < len(lines):
                remaining = '\n'.join(lines[i+1:]).strip()
                if remaining:
                    explanation = explanation + " " + remaining
            break
    
    # If no structured format, try to infer from text
    if label == "Unknown":
        content_lower = content.lower()
        if 'no error' in content_lower or 'is correct' in content_lower or 'correct' in content_lower:
            label = 'CORRECT'
        elif 'error' in content_lower or 'incorrect' in content_lower or 'mistake' in content_lower:
            label = 'INCORRECT'
    
    # If still no explanation, use the whole content
    if not explanation:
        explanation = content
    
    return thinking, label, explanation


def run_inference_medgemma(
    model,
    processor,
    test_df: pd.DataFrame,
    use_few_shot: bool = True,
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512
) -> List[Dict]:
    """
    Run inference on test data using MedGemma model.
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        test_df = test_df.head(max_samples)
    
    model.eval()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running MedGemma inference"):
        note = row['Text']
        ground_truth = row['Error Flag']  # 0 = Safe, 1 = Has Error
        error_type = row.get('Error Type', '')
        
        # Build prompt
        messages, _ = build_error_detection_prompt(note, use_few_shot, use_cot, MODEL_TYPE_MEDGEMMA)
        
        # Apply chat template for MedGemma
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_length = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None
            )
        
        # Decode output
        output_ids = generation[0][input_length:]
        content = processor.decode(output_ids, skip_special_tokens=True)
        
        # Extract label and explanation (no thinking for MedGemma)
        thinking, predicted_label, explanation = parse_response("", content)
        
        # Convert ground truth to label
        gt_label = "INCORRECT" if ground_truth == 1 else "CORRECT"
        
        # Check if prediction is correct
        correct = (predicted_label == gt_label)
        
        results.append({
            'text_id': row.get('Text ID', f'sample_{idx}'),
            'dataset': row.get('dataset', 'unknown'),
            'note': note,
            'ground_truth_flag': int(ground_truth),
            'ground_truth_label': gt_label,
            'error_type': error_type,
            'predicted_label': predicted_label,
            'explanation': explanation,
            'thinking': thinking,
            'correct': correct,
            'thinking_content': "",
            'final_content': content
        })
    
    return results


def run_inference_qwen(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    use_few_shot: bool = True,
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    thinking_budget: int = 1024
) -> List[Dict]:
    """
    Run inference on test data using Qwen3 model.
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        test_df = test_df.head(max_samples)
    
    model.eval()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running Qwen inference"):
        note = row['Text']
        ground_truth = row['Error Flag']  # 0 = Safe, 1 = Has Error
        error_type = row.get('Error Type', '')
        
        # Build prompt
        messages, enable_thinking = build_error_detection_prompt(note, use_few_shot, use_cot, MODEL_TYPE_QWEN)
        
        # Apply chat template with Qwen3's native thinking mode
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Use Qwen3's native CoT
        )
        
        # Tokenize input
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_length = model_inputs.input_ids.size(-1)
        
        # First generation up to thinking budget (official Qwen3 method)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=thinking_budget,
                temperature=temperature,
                top_p=0.95,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0, input_length:].tolist()
        
        # Check if generation finished or thinking budget reached
        if IM_END_TOKEN_ID not in output_ids:
            # Check if thinking process finished
            if THINK_END_TOKEN_ID not in output_ids:
                # Thinking budget reached - inject early stopping prompt
                early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                early_stopping_ids = tokenizer(
                    [early_stopping_text], 
                    return_tensors="pt", 
                    add_special_tokens=False
                ).input_ids.to(model.device)
                
                input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
            else:
                input_ids = generated_ids
            
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            
            # Second generation to complete the response
            remaining_tokens = max_new_tokens - (input_ids.size(-1) - input_length)
            if remaining_tokens > 0:
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=remaining_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
        
        # Parse using official Qwen3 method (token-based)
        thinking_content, content = parse_qwen3_output(
            tokenizer, 
            model_inputs.input_ids, 
            generated_ids
        )
        
        # Extract label and explanation
        thinking, predicted_label, explanation = parse_response(
            thinking_content, 
            content
        )
        
        # Convert ground truth to label
        gt_label = "INCORRECT" if ground_truth == 1 else "CORRECT"
        
        # Check if prediction is correct
        correct = (predicted_label == gt_label)
        
        results.append({
            'text_id': row.get('Text ID', f'sample_{idx}'),
            'dataset': row.get('dataset', 'unknown'),
            'note': note,
            'ground_truth_flag': int(ground_truth),
            'ground_truth_label': gt_label,
            'error_type': error_type,
            'predicted_label': predicted_label,
            'explanation': explanation,
            'thinking': thinking,
            'correct': correct,
            'thinking_content': thinking_content,
            'final_content': content
        })
    
    return results


def run_inference_gemma(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    use_few_shot: bool = True,
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512
) -> List[Dict]:
    """
    Run inference on test data using fine-tuned Gemma/MedGemma model (text-only).
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        test_df = test_df.head(max_samples)
    
    model.eval()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running Gemma inference"):
        note = row['Text']
        ground_truth = row['Error Flag']  # 0 = Safe, 1 = Has Error
        error_type = row.get('Error Type', '')
        
        # Build prompt (use generic format, not MedGemma multimodal format)
        messages, _ = build_error_detection_prompt(note, use_few_shot, use_cot, MODEL_TYPE_GENERIC)
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_length = model_inputs.input_ids.size(-1)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        output_ids = generated_ids[0][input_length:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Extract label and explanation (no thinking for Gemma)
        thinking, predicted_label, explanation = parse_response("", content)
        
        # Convert ground truth to label
        gt_label = "INCORRECT" if ground_truth == 1 else "CORRECT"
        
        # Check if prediction is correct
        correct = (predicted_label == gt_label)
        
        results.append({
            'text_id': row.get('Text ID', f'sample_{idx}'),
            'dataset': row.get('dataset', 'unknown'),
            'note': note,
            'ground_truth_flag': int(ground_truth),
            'ground_truth_label': gt_label,
            'error_type': error_type,
            'predicted_label': predicted_label,
            'explanation': explanation,
            'thinking': thinking,
            'correct': correct,
            'thinking_content': "",
            'final_content': content
        })
    
    return results


def run_inference(
    model,
    tokenizer_or_processor,
    test_df: pd.DataFrame,
    model_type: str,
    use_few_shot: bool = True,
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    thinking_budget: int = 1024
) -> List[Dict]:
    """
    Run inference on test data - dispatches to appropriate model-specific function.
    """
    if model_type == MODEL_TYPE_MEDGEMMA:
        return run_inference_medgemma(
            model=model,
            processor=tokenizer_or_processor,
            test_df=test_df,
            use_few_shot=use_few_shot,
            use_cot=use_cot,
            max_samples=max_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    elif model_type == MODEL_TYPE_GEMMA:
        return run_inference_gemma(
            model=model,
            tokenizer=tokenizer_or_processor,
            test_df=test_df,
            use_few_shot=use_few_shot,
            use_cot=use_cot,
            max_samples=max_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    else:
        return run_inference_qwen(
            model=model,
            tokenizer=tokenizer_or_processor,
            test_df=test_df,
            use_few_shot=use_few_shot,
            use_cot=use_cot,
            max_samples=max_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget
        )


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy, precision, recall, F1."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Binary classification metrics (CORRECT vs INCORRECT)
    tp = sum(1 for r in results if r['predicted_label'] == 'INCORRECT' and r['ground_truth_label'] == 'INCORRECT')
    fp = sum(1 for r in results if r['predicted_label'] == 'INCORRECT' and r['ground_truth_label'] == 'CORRECT')
    tn = sum(1 for r in results if r['predicted_label'] == 'CORRECT' and r['ground_truth_label'] == 'CORRECT')
    fn = sum(1 for r in results if r['predicted_label'] == 'CORRECT' and r['ground_truth_label'] == 'INCORRECT')
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Error Detection Inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (local or HuggingFace)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for this model in results (default: use model_path)")
    parser.add_argument("--model_type", type=str, default=None,
                       choices=["qwen", "medgemma", "gemma", "generic"],
                       help="Model type (auto-detected if not specified)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["ms", "uw", "all"],
                       help="Which test dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to test (default: all)")
    
    # Prompting arguments
    parser.add_argument("--no_few_shot", action="store_true",
                       help="Disable few-shot examples")
    parser.add_argument("--no_cot", action="store_true",
                       help="Disable chain-of-thought reasoning")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--thinking_budget", type=int, default=1024,
                       help="Thinking budget for Qwen3 (default: 1024)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/inference",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect or use specified model type
    model_type = args.model_type or detect_model_type(args.model_path)
    
    # Set model name
    model_name = args.model_name or args.model_path.replace('/', '_')
    
    print(f"\n{'='*60}")
    print(f"🔬 Medical Error Detection Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Model Type: {model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Few-shot: {not args.no_few_shot}")
    print(f"CoT: {not args.no_cot}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer_or_processor = load_model_and_tokenizer(args.model_path, model_type)
    
    # Load test data
    test_df = load_test_data(args.dataset)
    
    # Run inference
    results = run_inference(
        model=model,
        tokenizer_or_processor=tokenizer_or_processor,
        test_df=test_df,
        model_type=model_type,
        use_few_shot=not args.no_few_shot,
        use_cot=not args.no_cot,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['confusion_matrix']['true_positive']}")
    print(f"  FP: {metrics['confusion_matrix']['false_positive']}")
    print(f"  TN: {metrics['confusion_matrix']['true_negative']}")
    print(f"  FN: {metrics['confusion_matrix']['false_negative']}")
    print(f"{'='*60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{timestamp}_results.jsonl"
    )
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"✅ Detailed results saved to: {results_file}")
    
    # Save summary
    summary = {
        'model_path': args.model_path,
        'model_name': model_name,
        'dataset': args.dataset,
        'few_shot': not args.no_few_shot,
        'cot': not args.no_cot,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    summary_file = os.path.join(
        args.output_dir,
        f"{model_name}_{args.dataset}_{timestamp}_summary.json"
    )
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
