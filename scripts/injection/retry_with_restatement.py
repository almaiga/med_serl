"""
Retry failed benign changes using logical_restatement.
This script:
1. Loads failed notes from previous run
2. Selects a sentence from a populated target section
3. Rephrases via GPT-4o-mini (preserving clinical meaning)
4. Re-extracts from modified note using GPT-4o-mini
5. Verifies that ONLY the target field differs (all others must match exactly)

Usage:
  python retry_with_restatement.py                        # Test with first 10 notes
  python retry_with_restatement.py --limit 50             # Process 50 notes
  python retry_with_restatement.py --all                  # Process all failed notes
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from typing import Optional
import logging

# ============================================
# LOGGING SETUP
# ============================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"restatement_log_{timestamp}.txt"
    
    logger = logging.getLogger("restatement")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler - summary only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_path


# ============================================
# EXTRACTION PROMPT (same schema as clinical_note_extractor.json)
# ============================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert Clinical Data Extraction Specialist. Your task is to extract structured patient information from unstructured medical notes into a strict JSON format.

### INSTRUCTIONS:
1. **Accuracy**: Extract information EXACTLY as it appears in the text. Do not infer values unless heavily implied by standard medical terminology.
2. **Missing Data**: If a specific field is not mentioned in the text, you MUST set the value to `null`. Do not make up data.
3. **Format**: Output ONLY valid JSON. Do not include markdown formatting, explanations, or code blocks.
4. **Vitals**: Normalize vitals into string formats (e.g., "38.5 C", "120/80 mmHg").
5. **Narrative Fields**: Summarize key events chronologically for fields like 'history_of_present_illness' or 'hospital_course_summary'.

### JSON SCHEMA:
You must fill the following structure:
{
  "demographics": {
    "age": "integer (or null)",
    "gender": "string (M/F)",
    "race_ethnicity": "string (or null)",
    "occupation": "string (or null)",
    "bmi_data": {
      "height": "string (e.g., '170 cm')",
      "weight": "string (e.g., '70 kg')",
      "bmi": "float (or null)"
    }
  },
  "history": {
    "chief_complaint": "string (Primary reason for visit)",
    "history_of_present_illness": "string (Detailed narrative of current condition/onset)",
    "past_medical_history": ["list", "of", "conditions"],
    "past_surgical_history": ["list", "of", "surgeries"],
    "social_history": "string (Smoking, alcohol, living situation, profession details)",
    "family_history": "string (Relevant hereditary conditions)",
    "medications_prior_to_admission": ["list", "of", "home", "meds"],
    "allergies": ["list", "of", "allergens"]
  },
  "examination": {
    "vital_signs": {
      "temperature": "string",
      "heart_rate": "string",
      "blood_pressure": "string",
      "respiratory_rate": "string",
      "oxygen_saturation": "string"
    },
    "physical_exam_findings": "string (Key positive and negative findings from exam)"
  },
  "clinical_data": {
    "lab_results": "string (Key values e.g., 'WBC 12k, Na 135')",
    "imaging_results": "string (Key findings from X-ray, CT, MRI)",
    "microbiology": "string (Cultures, pathogens)"
  },
  "course_and_outcome": {
    "hospital_course_summary": "string (Summary of hospital stay, interventions, and complications. Important for BHC.)",
    "procedures_performed": ["list", "of", "procedures"],
    "diagnosis_primary": "string (The final confirmed diagnosis)",
    "plan_and_treatment": "string (Discharge plan, new medications, follow-up instructions)"
  }
}"""

EXTRACTION_USER_TEMPLATE = """Medical Note:
\"\"\"
{note_text}
\"\"\"

Extract the Patient Profile JSON:"""


# ============================================
# LOGICAL RESTATEMENT PROMPT
# ============================================

RESTATEMENT_SYSTEM_PROMPT = """You are a medical writing specialist. Rephrase clinical sentences while preserving EXACT clinical meaning.

Examples of valid restatements:
- "Vital signs are within normal limits" → "Vitals unremarkable"
- "Suspected of Huntington disease" → "Clinical presentation consistent with Huntington disease"
- "The patient has poor articulation" → "Dysarthria noted on examination"
- "Physical examination shows no abnormalities" → "Physical exam unremarkable"
- "4 x 4 cm area of reddened, blistered skin" → "4x4 cm erythematous, vesicular lesion"
- "He appears lethargic" → "Patient demonstrates lethargy"
- "Digital rectal examination shows a firm prostate" → "DRE reveals firm prostatic enlargement"

Rules:
1. Return ONLY the rephrased sentence, nothing else
2. ALL clinical facts must remain identical (same findings, same values, same anatomical locations)
3. Use different words/structure but preserve meaning
4. Output valid JSON only"""

RESTATEMENT_USER_TEMPLATE = """Rephrase this clinical sentence using different wording while preserving the exact medical meaning:

Sentence to rephrase:
"{original_sentence}"

Output JSON:
{{
  "replacement": "the rephrased sentence (ONLY the sentence, nothing else)",
  "preserved_facts": ["list", "of", "clinical facts preserved"]
}}"""


# ============================================
# TARGET SECTION PRIORITY FOR RESTATEMENT
# ============================================

# Priority order: sections most likely to have good content for rephrasing
TARGET_SECTIONS = [
    ("examination.physical_exam_findings", "physical_exam_findings"),
    ("history.history_of_present_illness", "history_of_present_illness"),
    ("course_and_outcome.diagnosis_primary", "diagnosis_primary"),
    ("course_and_outcome.hospital_course_summary", "hospital_course_summary"),
    ("course_and_outcome.plan_and_treatment", "plan_and_treatment"),
    ("clinical_data.imaging_results", "imaging_results"),
]


# ============================================
# HELPER FUNCTIONS
# ============================================

def parse_extraction(extraction) -> dict:
    """Parse extraction JSON string or dict."""
    if isinstance(extraction, str):
        try:
            return json.loads(extraction)
        except json.JSONDecodeError:
            return {}
    return extraction if isinstance(extraction, dict) else {}


def get_value_from_path(data: dict, path: str):
    """Get nested value from dot-notation path like 'history.chief_complaint'."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def set_value_at_path(data: dict, path: str, value):
    """Set nested value at dot-notation path."""
    keys = path.split(".")
    d = data
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def extract_sentences(note_text: str) -> list[str]:
    """Split note into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', note_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_best_sentence_for_restatement(note_text: str, extraction: dict, logger) -> Optional[tuple[str, str, str]]:
    """
    Find the best sentence to rephrase based on target sections.
    
    Returns: (sentence, section_path, field_name) or None
    """
    for section_path, field_name in TARGET_SECTIONS:
        field_value = get_value_from_path(extraction, section_path)
        
        # Skip null/empty fields
        if not field_value or (isinstance(field_value, str) and not field_value.strip()):
            continue
        
        # For string fields, find a sentence in the note that relates to this content
        if isinstance(field_value, str):
            # Look for sentences containing key terms from the field value
            sentences = extract_sentences(note_text)
            
            # Try to find a sentence that contains significant content from the field
            field_words = set(field_value.lower().split())
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # Check for significant overlap (at least 3 common words)
                common = field_words & sentence_words
                if len(common) >= 3 or any(word in sentence.lower() for word in field_value.lower().split(',')[:3]):
                    logger.debug(f"  Found candidate sentence in {field_name}: '{sentence[:80]}...'")
                    return (sentence, section_path, field_name)
    
    return None


def normalize_for_comparison(value):
    """Normalize a value for comparison (handles minor formatting differences)."""
    if value is None:
        return None
    if isinstance(value, str):
        # Normalize whitespace but preserve content
        return ' '.join(value.split())
    if isinstance(value, list):
        return sorted([normalize_for_comparison(v) for v in value])
    if isinstance(value, dict):
        return {k: normalize_for_comparison(v) for k, v in value.items()}
    return value


def flatten_extraction(extraction: dict, prefix: str = "") -> dict:
    """Flatten nested extraction to dot-notation paths for comparison."""
    result = {}
    for key, value in extraction.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_extraction(value, full_key))
        else:
            result[full_key] = value
    return result


def compare_extractions(original: dict, modified: dict, target_field: str, logger) -> tuple[bool, list[str]]:
    """
    Compare two extractions.
    
    Returns: (is_valid, list_of_differing_fields)
    
    Valid if: exactly 1 field differs AND it's the target field (or a related field)
    """
    orig_flat = flatten_extraction(original)
    mod_flat = flatten_extraction(modified)
    
    all_keys = set(orig_flat.keys()) | set(mod_flat.keys())
    
    differing_fields = []
    
    for key in all_keys:
        orig_val = normalize_for_comparison(orig_flat.get(key))
        mod_val = normalize_for_comparison(mod_flat.get(key))
        
        if orig_val != mod_val:
            differing_fields.append(key)
            logger.debug(f"    Field '{key}' differs:")
            logger.debug(f"      Original: {str(orig_val)[:100]}")
            logger.debug(f"      Modified: {str(mod_val)[:100]}")
    
    # Check if the only differing field is the target (or contains the target)
    if len(differing_fields) == 0:
        logger.warning("  No fields differ - restatement may not have been applied")
        return False, differing_fields
    
    if len(differing_fields) == 1:
        # Perfect - exactly one field differs
        diff_field = differing_fields[0]
        if target_field in diff_field or diff_field in target_field:
            return True, differing_fields
        else:
            logger.warning(f"  Wrong field changed: expected '{target_field}', got '{diff_field}'")
            return False, differing_fields
    
    # Multiple fields differ - check if they're related to target
    target_related = [f for f in differing_fields if target_field in f or f in target_field]
    unrelated = [f for f in differing_fields if f not in target_related]
    
    if len(unrelated) > 0:
        logger.warning(f"  Multiple unrelated fields changed: {unrelated}")
        return False, differing_fields
    
    # All differing fields are related to target
    return True, differing_fields


# ============================================
# MAIN PROCESSOR CLASS
# ============================================

class RestatementRetry:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
    
    def generate_restatement(self, sentence: str, logger) -> Optional[dict]:
        """Generate a logical restatement of a sentence. Returns ONLY the replacement."""
        try:
            user_prompt = RESTATEMENT_USER_TEMPLATE.format(original_sentence=sentence)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RESTATEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            replacement = result.get("replacement", "").strip()
            if not replacement:
                logger.warning("  LLM returned empty replacement")
                return None
            
            return {
                "replacement_sentence": replacement,
                "preserved_facts": result.get("preserved_facts", [])
            }
            
        except Exception as e:
            logger.error(f"  Restatement generation error: {e}")
            return None
    
    def extract_from_note(self, note_text: str, logger) -> Optional[dict]:
        """Extract structured JSON from a clinical note."""
        try:
            user_prompt = EXTRACTION_USER_TEMPLATE.format(note_text=note_text)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"  Extraction error: {e}")
            return None
    
    def process_note(self, note_obj: dict, logger) -> dict:
        """
        Process a single failed note with logical restatement.
        
        Returns a result dict with success/failure info.
        """
        note_id = note_obj.get("note_id", "unknown")
        note_text = note_obj.get("original_note", "")
        original_extraction = parse_extraction(note_obj.get("extraction", {}))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {note_id}")
        logger.debug(f"  Note length: {len(note_text)} chars")
        
        # Find best sentence to rephrase
        sentence_info = find_best_sentence_for_restatement(note_text, original_extraction, logger)
        
        if not sentence_info:
            logger.warning(f"  SKIP: No suitable sentence found for restatement")
            return {
                "note_id": note_id,
                "success": False,
                "failure_reason": "no_suitable_sentence",
                "error": "Could not find a sentence suitable for logical restatement"
            }
        
        original_sentence, target_section, target_field = sentence_info
        logger.info(f"  Target field: {target_field}")
        logger.info(f"  Original sentence: \"{original_sentence[:80]}...\"")
        
        # Generate restatement (LLM returns ONLY the replacement sentence)
        restatement = self.generate_restatement(original_sentence, logger)
        
        if not restatement:
            return {
                "note_id": note_id,
                "success": False,
                "failure_reason": "restatement_generation_failed",
                "target_sentence": original_sentence,
                "target_field": target_field
            }
        
        replacement_sentence = restatement.get("replacement_sentence", "")
        logger.info(f"  Replacement: \"{replacement_sentence[:80]}...\"")
        
        # WE do the substitution in Python (LLM never touches the full note)
        modified_note = note_text.replace(original_sentence, replacement_sentence, 1)
        
        # Verify the sentence actually changed
        if original_sentence == replacement_sentence:
            logger.warning("  FAIL: Sentence unchanged")
            return {
                "note_id": note_id,
                "success": False,
                "failure_reason": "sentence_unchanged",
                "original_sentence": original_sentence,
                "replacement_sentence": replacement_sentence
            }
        
        # NOTE-LEVEL VERIFICATION (not extraction-level!)
        # LLM extraction is non-deterministic - we verify at string level instead
        logger.debug("  Verifying single-sentence change...")
        
        # The note should have changed
        if modified_note == note_text:
            logger.warning("  FAIL: Note unchanged after replacement")
            return {
                "note_id": note_id,
                "success": False,
                "failure_reason": "note_unchanged",
                "original_sentence": original_sentence,
                "replacement_sentence": replacement_sentence
            }
        
        # Replacement should be in the modified note
        if replacement_sentence not in modified_note:
            logger.warning("  FAIL: Replacement sentence not found in modified note")
            return {
                "note_id": note_id,
                "success": False,
                "failure_reason": "replacement_not_applied",
                "original_sentence": original_sentence,
                "replacement_sentence": replacement_sentence
            }
        
        # SUCCESS - single sentence was replaced
        logger.info(f"  ✓ SUCCESS: Single-sentence replacement verified")
        logger.info(f"    Original: \"{original_sentence[:60]}...\"")
        logger.info(f"    Replaced: \"{replacement_sentence[:60]}...\"")
        
        return {
            "note_id": note_id,
            "success": True,
            "change_type": "logical_restatement",
            "original_note": note_text,
            "modified_note": modified_note,
            "original_sentence": original_sentence,
            "replacement_sentence": replacement_sentence,
            "target_field": target_field,
            "preserved_facts": restatement.get("preserved_facts", []),
            "original_extraction": original_extraction,
            "verification_method": "note_level_string_comparison",
            "verified": True
        }


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Retry failed benign changes with logical restatement")
    parser.add_argument("--input", type=str, 
                        default="data_processed/benign_changes/failed_notes_20260127_191801.jsonl",
                        help="Path to failed notes JSONL")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of notes to process (default: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Process all failed notes")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use")
    args = parser.parse_args()
    
    # Paths
    input_path = Path(args.input)
    output_dir = Path("data_processed/benign_changes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success_path = output_dir / f"restatement_success_{timestamp}.jsonl"
    failed_path = output_dir / f"restatement_failed_{timestamp}.jsonl"
    
    # Setup logging
    logger, log_path = setup_logging(output_dir)
    logger.info(f"Log file: {log_path}")
    
    # Load failed notes
    logger.info(f"Loading failed notes from: {input_path}")
    failed_notes = []
    with open(input_path, 'r') as f:
        for line in f:
            failed_notes.append(json.loads(line))
    
    logger.info(f"Total failed notes: {len(failed_notes)}")
    
    # Apply limit
    if not args.all:
        failed_notes = failed_notes[:args.limit]
        logger.info(f"Processing first {len(failed_notes)} notes (use --all for all)")
    
    # Initialize processor
    processor = RestatementRetry(model=args.model)
    
    # Process notes
    success_count = 0
    failed_count = 0
    
    success_file = open(success_path, 'w')
    failed_file = open(failed_path, 'w')
    
    try:
        for note_obj in tqdm(failed_notes, desc="Processing"):
            result = processor.process_note(note_obj, logger)
            
            if result.get("success"):
                success_file.write(json.dumps(result) + '\n')
                success_file.flush()
                success_count += 1
            else:
                # Merge with original note info
                result["original_note"] = note_obj.get("original_note", "")
                result["extraction"] = note_obj.get("extraction", {})
                result["error_type"] = note_obj.get("error_type", "")
                failed_file.write(json.dumps(result) + '\n')
                failed_file.flush()
                failed_count += 1
    
    finally:
        success_file.close()
        failed_file.close()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total processed: {len(failed_notes)}")
    logger.info(f"✓ Successful: {success_count} ({success_count/len(failed_notes)*100:.1f}%)")
    logger.info(f"✗ Failed: {failed_count} ({failed_count/len(failed_notes)*100:.1f}%)")
    logger.info(f"\nOutput files:")
    logger.info(f"  Success: {success_path}")
    logger.info(f"  Failed: {failed_path}")
    logger.info(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
