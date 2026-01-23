import json
import random
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import Optional
from medical_knowledge_base import get_verified_synonym, get_temporal_synonym, RxNormClient

# ============================================
# PROMPT LOADER
# ============================================

def load_prompt_configs(config_path: Path) -> dict:
    """Load all benign change prompt configs from single JSON file (array format)."""
    with open(config_path, 'r') as f:
        content = f.read()
        # Remove comment lines starting with //
        lines = [line for line in content.split('\n') if not line.strip().startswith('//')]
        clean_content = '\n'.join(lines)
        prompts_list = json.loads(clean_content)
    
    # Convert list to dict keyed by change_type
    configs = {}
    for prompt in prompts_list:
        configs[prompt["change_type"]] = prompt
    
    return configs


# ============================================
# EXTRACTION PARSER
# ============================================

def parse_extraction(extraction: str | dict) -> dict:
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


# ============================================
# SENTENCE EXTRACTION
# ============================================

def extract_sentences(note_text: str) -> list[str]:
    """Split note into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', note_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_sentence_with_term(note_text: str, term: str) -> Optional[tuple[str, int, int]]:
    """Find the sentence containing the term and its position in the note.
    
    Returns: (sentence, start_pos, end_pos) or None
    """
    if not term or term not in note_text:
        return None
    
    sentences = extract_sentences(note_text)
    
    for sentence in sentences:
        if term in sentence:
            # Find position in original note
            start_pos = note_text.find(sentence)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                return (sentence, start_pos, end_pos)
    
    return None


def replace_sentence_in_note(note_text: str, old_sentence: str, new_sentence: str) -> str:
    """Replace a single sentence in the note, leaving everything else unchanged."""
    # Use replace with count=1 to only replace first occurrence
    return note_text.replace(old_sentence, new_sentence, 1)


# ============================================
# TERM EXTRACTION FROM EXTRACTION
# ============================================

def extract_target_term(extraction: dict, target_sections: list, note_text: str) -> Optional[str]:
    """Extract a target term from the specified sections of the extraction."""
    
    for section_path in target_sections:
        value = get_value_from_path(extraction, section_path)
        
        if value is None:
            continue
        
        if isinstance(value, list) and len(value) > 0:
            # Try each term in the list
            random.shuffle(value)
            for term in value:
                if term and term in note_text:
                    return term
        elif isinstance(value, str) and value:
            if value in note_text:
                return value
            terms = [t.strip() for t in value.split(",")]
            for term in terms:
                if term and len(term) > 3 and term in note_text:
                    return term
    
    return None


def extract_temporal_term(note_text: str) -> Optional[str]:
    """Extract a time expression from the note using regex."""
    patterns = [
        r'\d+-(?:year|month|week|day|hour)\s+history',
        r'(?:past|last|over the past|for the past)\s+\d+\s+(?:years?|months?|weeks?|days?|hours?)',
        r'\d+\s+(?:years?|months?|weeks?|days?|hours?)\s+ago',
        r'(?:past|last)\s+(?:few|several|couple of)\s+(?:years?|months?|weeks?|days?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None


def extract_irrelevant_term(note_text: str) -> Optional[str]:
    """Extract a non-clinical detail from the note."""
    relationship_patterns = [
        r'(?:by (?:her|his) )(husband|wife|mother|father|son|daughter|family member)',
        r'(?:brought in by )([\w\s]+?)(?=\s+because|\s+for|\s+due)',
    ]
    
    for pattern in relationship_patterns:
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


# ============================================
# BENIGN CHANGE GENERATOR (No Ollama)
# ============================================

class BenignChangeGenerator:
    def __init__(self, config_path: Path, model: str = "gpt-4o-mini", use_local_verifier: bool = False):
        self.client = OpenAI()
        self.model = model
        self.configs = load_prompt_configs(config_path)
        self.change_types = list(self.configs.keys())
        
        # API clients (no auth needed)
        self.rxnorm = RxNormClient()
        
        # No local verifier for now
        self.verifier = None
    
    # Map note type (error_type) to allowed change types
    NOTE_TYPE_TO_CHANGE_TYPES = {
        "diagnosis": ["pseudo_factual", "temporal_rephrasing"],
        "pharmacotherapy": ["equivalent_citation", "pseudo_factual"],
        "treatment": ["equivalent_citation", "pseudo_factual", "temporal_rephrasing"],
        "management": ["pseudo_factual", "temporal_rephrasing", "equivalent_citation"],
        "causalorganism": ["pseudo_factual"],
        # fallback: all types if not specified
    }

    # Map error_type to preferred extraction fields for target term selection
    ERROR_TYPE_TO_FIELDS = {
        "diagnosis": [
            "course_and_outcome.diagnosis_primary",
            "history.past_medical_history",
            "history.chief_complaint",
            "history.history_of_present_illness",
            "examination.physical_exam_findings"
        ],
        "pharmacotherapy": [
            "history.medications_prior_to_admission",
            "course_and_outcome.plan_and_treatment",
            "course_and_outcome.procedures_performed"
        ],
        "treatment": [
            "history.medications_prior_to_admission",
            "course_and_outcome.plan_and_treatment",
            "course_and_outcome.procedures_performed"
        ],
        "management": [
            "history.medications_prior_to_admission",
            "course_and_outcome.plan_and_treatment",
            "course_and_outcome.procedures_performed",
            "history.past_medical_history"
        ],
        "causalorganism": [
            "clinical_data.microbiology"
        ],
        # fallback: []
    }

    def get_allowed_change_types(self, note_type: str) -> list:
        """Return allowed change types for a given note type."""
        return self.NOTE_TYPE_TO_CHANGE_TYPES.get(note_type, self.change_types)
    
    def get_verified_replacement(self, term: str, change_type: str) -> Optional[dict]:
        """Get a verified replacement from FREE medical APIs."""
        
        if change_type == "pseudo_factual":
            result = get_verified_synonym(term, verbose=True)  # Enable verbose for debugging
            if result.get("verified") and result.get("synonyms"):
                return {
                    "original": term,
                    "replacement": random.choice(result["synonyms"]),
                    "source": result["source"],
                    "verified": True
                }
        
        elif change_type == "temporal_rephrasing":
            result = get_temporal_synonym(term)
            if result.get("verified") and result.get("replacement"):
                return {
                    "original": term,
                    "replacement": result["replacement"],
                    "source": result["source"],
                    "verified": True
                }
        
        elif change_type == "equivalent_citation":
            drug_info = self.rxnorm.get_drug_class(term)
            if drug_info and drug_info.get("classes"):
                return {
                    "original": term,
                    "replacement": drug_info["classes"][0],
                    "source": "RxNorm",
                    "verified": True
                }
        
        return None
    
    def find_target_term(self, note_text: str, extraction: dict, change_type: str, note_type: str = None) -> Optional[str]:
        """
        Find a suitable target term based on change type, extraction, and note type.
        Prioritize fields based on error_type if available.
        """
        # Determine note_type (error_type)
        if note_type is None:
            note_type = extraction.get("error_type") or extraction.get("note_type") or ""
        # Use preferred fields for this note type
        preferred_fields = self.ERROR_TYPE_TO_FIELDS.get(note_type, [])

        config = self.configs[change_type]
        target_sections = config.get("target_sections", [])

        # For temporal_rephrasing, use regex
        if change_type == "temporal_rephrasing":
            return extract_temporal_term(note_text)

        # For irrelevant_correlation, use regex
        if change_type == "irrelevant_correlation":
            term = extract_irrelevant_term(note_text)
            if term:
                return term

        # Try preferred fields first
        if preferred_fields:
            term = extract_target_term(extraction, preferred_fields, note_text)
            if term:
                return term

        # Fallback to config target_sections
        return extract_target_term(extraction, target_sections, note_text)
    
    def assign_change_types(self, notes: list) -> list:
        """Assign each note a compatible change type based on its note type."""
        assignments = []
        for note in notes:
            note_type = note.get("error_type", "")
            allowed_types = self.get_allowed_change_types(note_type)
            # Pick a random allowed change type for this note
            if allowed_types:
                change_type = random.choice(allowed_types)
                assignments.append({"note": note, "change_type": change_type})
        return assignments
    
    def generate_change(self, note_text: str, change_type: str, extraction: dict, allow_fallback: bool = False, tried_types=None, note_type: str = None) -> dict:
        """Generate a specific type of benign change (SINGLE SENTENCE ONLY). If fails, optionally try another compatible type."""
        if tried_types is None:
            tried_types = set()
        tried_types.add(change_type)

        # Determine allowed change types for this note
        if note_type is None:
            note_type = extraction.get("error_type") or extraction.get("note_type") or ""
        allowed_types = self.get_allowed_change_types(note_type)

        config = self.configs[change_type]
        target_term = self.find_target_term(note_text, extraction, change_type, note_type=note_type)
        
        # Find the sentence containing the target term
        sentence_info = find_sentence_with_term(note_text, target_term)
        original_sentence, start_pos, end_pos = None, None, None
        if sentence_info:
            original_sentence, start_pos, end_pos = sentence_info
        elif change_type == "logical_restatement":
            # Fallback: use corrected_sentence or first non-empty sentence
            original_sentence = extraction.get("corrected_sentence")
            if not original_sentence:
                sentences = extract_sentences(note_text)
                original_sentence = next((s for s in sentences if s.strip()), None)
            if not original_sentence or not original_sentence.strip():
                error_msg = f"Could not isolate sentence for logical_restatement (empty original_sentence)"
                print(f"[DEBUG] logical_restatement: original_sentence is empty for note_id={extraction.get('note_id', 'N/A')}")
                if allow_fallback:
                    fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                    if fallback_type:
                        return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
                return {
                    "change_type": change_type,
                    "original_note": note_text,
                    "modified_note": note_text,
                    "change_made": False,
                    "is_benign": True,
                    "api_success": False,
                    "error": error_msg
                }
            start_pos = note_text.find(original_sentence)
            end_pos = start_pos + len(original_sentence) if start_pos != -1 else None
        else:
            if not sentence_info:
                error_msg = f"Could not isolate sentence containing '{target_term}'"
                if allow_fallback:
                    fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                    if fallback_type:
                        return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
                return {
                    "change_type": change_type,
                    "original_note": note_text,
                    "modified_note": note_text,
                    "change_made": False,
                    "is_benign": True,
                    "api_success": False,
                    "error": error_msg
                }

        # TRY VERIFIED API SOURCE FIRST
        verified = self.get_verified_replacement(target_term, change_type)
        if verified and verified.get("verified"):
            modified_sentence = original_sentence.replace(target_term, verified["replacement"]) if target_term in original_sentence else original_sentence
            modified_note = replace_sentence_in_note(note_text, original_sentence, modified_sentence)
            return {
                "change_type": change_type,
                "original_note": note_text,
                "modified_note": modified_note,
                "original_sentence": original_sentence,
                "modified_sentence": modified_sentence,
                "original_term": target_term,
                "replacement_term": verified["replacement"] if target_term in original_sentence else "",
                "sentence_position": {"start": start_pos, "end": end_pos} if start_pos is not None else None,
                "is_benign": True,
                "change_made": True,
                "api_success": True,
                "verification_source": verified["source"],
                "verified": True
            }
        
        # FALLBACK TO LLM
        try:
            user_template = config["user_template"]
            required_vars = re.findall(r"\{(\w+)\}", user_template)
            format_kwargs = {"target_term": target_term, "note_text": note_text}
            if change_type == "logical_restatement":
                format_kwargs["original_sentence"] = original_sentence
                print(f"[DEBUG] logical_restatement LLM call for note_id={extraction.get('note_id', 'N/A')}")
                print(f"  original_sentence: {repr(original_sentence)}")
                print(f"  note_text: {repr(note_text[:120])}...")
                print(f"  user_template: {user_template}")
            for var in required_vars:
                if var not in format_kwargs:
                    val = extraction.get(var)
                    if val:
                        format_kwargs[var] = val
            missing_vars = [var for var in required_vars if var not in format_kwargs]
            if missing_vars:
                print(f"[DEBUG] logical_restatement missing_vars: {missing_vars}")
                if allow_fallback:
                    fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                    if fallback_type:
                        return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
                return {
                    "change_type": change_type,
                    "original_note": note_text,
                    "modified_note": note_text,
                    "change_made": False,
                    "is_benign": True,
                    "api_success": False,
                    "error": f"Missing required variables for prompt: {missing_vars}"
                }
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": user_template.format(**format_kwargs)}
                ],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            print(f"[DEBUG] logical_restatement LLM response: {response.choices[0].message.content[:200]}...")

            result = json.loads(response.choices[0].message.content)

            # --- Robust validation for logical_restatement ---
            if change_type == "logical_restatement":
                # Must have replacement_sentence and modified_note and change_made True
                if (
                    not isinstance(result, dict)
                    or not result.get("change_made")
                    or not result.get("replacement_sentence")
                    or not result.get("modified_note")
                    or result.get("replacement_sentence").strip() == original_sentence.strip()
                ):
                    print(f"[DEBUG] logical_restatement: LLM output incomplete or not a true restatement, falling back.")
                    if allow_fallback:
                        fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                        if fallback_type:
                            return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
                    return {
                        "change_type": change_type,
                        "original_note": note_text,
                        "modified_note": note_text,
                        "change_made": False,
                        "is_benign": True,
                        "api_success": False,
                        "error": "LLM did not produce a valid logical restatement"
                    }

            if not result.get("change_made"):
                if allow_fallback:
                    fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                    if fallback_type:
                        return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
                return {
                    "change_type": change_type,
                    "original_note": note_text,
                    "modified_note": note_text,
                    "change_made": False,
                    "is_benign": True,
                    "api_success": True,
                    "error": "LLM declined to make change"
                }
            
            modified_note = result.get("modified_note", note_text)
            # For logical_restatement, set modified_sentence for reporting
            modified_sentence = result.get("replacement_sentence", result.get("replacement_statement", ""))
            
            return {
                "change_type": change_type,
                "original_note": note_text,
                "modified_note": modified_note,
                "original_sentence": original_sentence,
                "modified_sentence": modified_sentence,
                "target_term": target_term,
                "replacement_term": result.get("replacement_term", result.get("replacement_expression", result.get("replacement_detail", result.get("replacement_statement", "")))),
                "sentence_position": {"start": start_pos, "end": end_pos} if start_pos is not None else None,
                "is_benign": True,
                "change_made": True,
                "api_success": True,
                "verified": False,
                "verification_source": "LLM (unverified)"
            }
            
        except Exception as e:
            print(f"[DEBUG] logical_restatement LLM exception: {e}")
            if allow_fallback:
                fallback_type = next((ct for ct in allowed_types if ct not in tried_types), None)
                if fallback_type:
                    return self.generate_change(note_text, fallback_type, extraction, allow_fallback=True, tried_types=tried_types, note_type=note_type)
            return {
                "change_type": change_type,
                "original_note": note_text,
                "modified_note": note_text,
                "change_made": False,
                "is_benign": True,
                "api_success": False,
                "error": str(e)
            }
    
    def generate_batch(self, notes: list, output_path: Path) -> dict:
        """Generate benign changes for all notes (ONE SENTENCE PER NOTE)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        assignments = self.assign_change_types(notes)
        random.shuffle(assignments)
        
        stats = {ct: {"success": 0, "failed": 0, "no_target": 0, "verified": 0, "llm": 0} for ct in self.change_types}
        
        with open(output_path, 'w') as f:
            for item in tqdm(assignments, desc="Generating benign changes"):
                note_obj = item["note"]
                change_type = item["change_type"]
                note_text = note_obj.get("correct_note", "")
                extraction = parse_extraction(note_obj.get("extraction", {}))
                
                # Enable fallback for each note
                result = self.generate_change(note_text, change_type, extraction, allow_fallback=True)
                
                result["note_id"] = note_obj.get("note_id")
                result["error_type"] = note_obj.get("error_type")
                result["corrected_sentence"] = note_obj.get("corrected_sentence")
                
                if result.get("change_made", False):
                    stats[result["change_type"]]["success"] += 1
                    if result.get("verified"):
                        stats[result["change_type"]]["verified"] += 1
                    else:
                        stats[result["change_type"]]["llm"] += 1
                elif "No suitable target" in result.get("error", ""):
                    stats[result["change_type"]]["no_target"] += 1
                else:
                    stats[result["change_type"]]["failed"] += 1
                
                f.write(json.dumps(result) + '\n')
        
        return stats


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    config_dir = Path("configs/prompts/benign_change_prompt.json")
    data_path = Path("data_processed/parsed_medical_note/extractions.jsonl")
    output_path = Path("data_processed/benign_changes/benign_train.jsonl")
    
    print("=" * 60)
    print("BENIGN CHANGE INJECTION (Single Sentence Modification)")
    print("=" * 60)
    
    print("\nLoading notes with extractions...")
    notes = []
    with open(data_path, 'r') as f:
        for line in f:
            notes.append(json.loads(line))
    
    print(f"Loaded {len(notes)} notes")
    
    generator = BenignChangeGenerator(config_dir, model="gpt-4o-mini", use_local_verifier=True)
    
    print(f"\nEach change type will get ~{len(notes)//len(generator.change_types)} notes")
    print("Change types loaded:")
    for i, ct in enumerate(generator.change_types):
        print(f"  {i+1}. {ct}: {generator.configs[ct]['description']}")
    
    print("\n⚠️  IMPORTANT: Only ONE SENTENCE will be modified per note.")
    print("    The rest of the note remains UNCHANGED.\n")
    
    stats = generator.generate_batch(notes, output_path)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print("\nGeneration Statistics:")
    total_success = total_failed = total_no_target = total_verified = total_llm = 0
    
    for ct, counts in stats.items():
        print(f"  {ct}:")
        print(f"    ✓ Success:   {counts['success']} (verified: {counts['verified']}, LLM: {counts['llm']})")
        print(f"    ✗ Failed:    {counts['failed']}")
        print(f"    ○ No target: {counts['no_target']}")
        total_success += counts['success']
        total_failed += counts['failed']
        total_no_target += counts['no_target']
        total_verified += counts['verified']
        total_llm += counts['llm']
    
    total = total_success + total_failed + total_no_target
    print(f"\nTotal: {total_success} success ({total_verified} verified, {total_llm} LLM), {total_failed} failed, {total_no_target} no target")
    if total > 0:
        print(f"Success rate: {total_success/total*100:.1f}%")
        if total_success > 0:
            print(f"Verification rate: {total_verified/total_success*100:.1f}%")