import json
import random
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import Optional

# Unified imports from medical_knowledge_base
from medical_knowledge_base import (
    get_verified_replacement as get_unified_replacement,
    get_pseudo_factual_replacement,
    get_pseudo_factual_replacement_with_fallback,
    get_temporal_replacement,
    get_equivalent_citation_replacement,
    RxNormLookup,
    VerifiedReplacement,
)

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

def clean_extracted_term(term: str) -> str:
    """Clean extracted term by removing prepositions, articles, and common verbs."""
    if not term:
        return term
    
    # Remove "Treatment with X" → "X"
    term = re.sub(r'^Treatment\s+with\s+', '', term, flags=re.IGNORECASE)
    term = re.sub(r'^Therapy\s+with\s+', '', term, flags=re.IGNORECASE)
    
    # Remove leading prepositions
    term = re.sub(r'^(via|by|with|from|through|of|for|in|on|at|to)\s+', '', term, flags=re.IGNORECASE)
    
    # Remove leading articles
    term = re.sub(r'^(the|a|an)\s+', '', term, flags=re.IGNORECASE)
    
    # Remove trailing verbs and verb phrases
    term = re.sub(r'\s+(is|was|are|were|be|been|being|prescribed|recommended|initiated|started|administered|suggested|advised|ordered)$', '', term, flags=re.IGNORECASE)
    
    # Remove leading verbs
    term = re.sub(r'^(is|was|are|were|be|been|being|prescribed|recommended|initiated|started|administered|suggested|advised|ordered)\s+', '', term, flags=re.IGNORECASE)
    
    # Strip "as needed", "daily", dosage info for medications
    term = re.sub(r'\s+(?:as needed|prn|daily|twice daily|bid|tid|qid|qhs|qd).*$', '', term, flags=re.IGNORECASE)
    term = re.sub(r'\s+(?:inhaler|tablet|capsule|injection|solution|cream|ointment|patch|spray)s?.*$', '', term, flags=re.IGNORECASE)
    
    return term.strip()


def is_valid_medical_term(term: str) -> bool:
    """Validate that term is likely a medical term, not a verb or generic word."""
    if not term or len(term) < 3:
        return False
    
    term_lower = term.lower().strip()
    
    # Reject common verbs that shouldn't be extracted as medical terms
    VERB_BLACKLIST = [
        "is", "was", "are", "were", "be", "been", "being",
        "prescribed", "recommended", "initiated", "started", 
        "administered", "suggested", "advised", "ordered",
        "given", "taken", "used", "received", "treated",
        "performed", "completed", "conducted", "done", "showed",
        "obtained", "revealed", "demonstrated",
    ]
    
    if term_lower in VERB_BLACKLIST:
        return False
    
    # ========== NEW: Reject lab values and partial numbers ==========
    
    # Reject terms that are mostly numbers/units (lab values like "000/mm3", "9000/mm3")
    if re.match(r'^\d+[,/]?\d*\s*(?:mm3|/mm3|mg|ml|g|dl|L|mEq|hpf|%)?$', term, re.IGNORECASE):
        return False
    
    # Reject terms starting with numbers followed by units (partial lab values)
    if re.match(r'^\d{2,}[,/]', term):  # "000/mm3", "9000/mm3"
        return False
    
    # Reject pure number patterns
    if re.match(r'^[\d,./\-\s]+$', term):
        return False
    
    # Reject terms that are lab value fragments
    LAB_VALUE_PATTERNS = [
        r'^\d+/mm',  # "9000/mm3"
        r'^mm\d',    # "mm3"
        r'^\d+%$',   # "52%"
        r'^\d+\s*(?:mg|ml|g|dl|mEq|hpf)',  # "135 U/L"
    ]
    for pattern in LAB_VALUE_PATTERNS:
        if re.match(pattern, term, re.IGNORECASE):
            return False
    
    # ========== END NEW ==========
    
    # Reject phrases with "and" that are likely not medical terms
    if " and " in term_lower:
        # "and mother", "father and brother" etc.
        family_words = ["mother", "father", "brother", "sister", "parent", "child", "family"]
        if any(word in term_lower for word in family_words):
            return False
    
    # Reject generic procedure/imaging terms that aren't specific diagnoses or drugs
    GENERIC_PROCEDURE_TERMS = [
        "series", "study", "contrast", "imaging", "scan", "test",
        "exam", "examination", "procedure", "assessment", "evaluation",
        "contrast series", "imaging study", "diagnostic test",
    ]
    if term_lower in GENERIC_PROCEDURE_TERMS:
        return False
    
    # Reject terms that are purely prepositions or conjunctions
    PREPOSITION_WORDS = ["and", "or", "but", "with", "from", "to", "in", "on", "at", "by", "for"]
    if term_lower in PREPOSITION_WORDS:
        return False
    
    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', term):
        return False
    
    return True


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
                if not term:
                    continue
                
                # Clean the term first
                cleaned_term = clean_extracted_term(term)
                
                # Validate it's a medical term
                if not is_valid_medical_term(cleaned_term):
                    continue
                
                # For long terms, try to extract shorter medical term
                if len(cleaned_term.split()) > 4:
                    shorter = _extract_shortest_medical_term(cleaned_term, note_text)
                    if shorter and is_valid_medical_term(shorter) and shorter in note_text:
                        return shorter
                
                # Check if cleaned term is in note
                if cleaned_term in note_text:
                    return cleaned_term
                
                # Fallback to original if cleaning broke it
                if term in note_text:
                    return term
        elif isinstance(value, str) and value:
            # Clean the value first
            cleaned_value = clean_extracted_term(value)
            
            # Validate it's a medical term
            if is_valid_medical_term(cleaned_value):
                # For long compound descriptions, extract shorter term
                if len(cleaned_value.split()) > 4:
                    shorter = _extract_shortest_medical_term(cleaned_value, note_text)
                    if shorter and is_valid_medical_term(shorter) and shorter in note_text:
                        return shorter
                
                # First try the cleaned value
                if cleaned_value in note_text:
                    return cleaned_value
            
            # Then try splitting on commas and picking shorter terms
            terms = [clean_extracted_term(t.strip()) for t in value.split(",")]
            # Sort by length (prefer shorter, more specific terms)
            terms.sort(key=len)
            for term in terms:
                if term and len(term) > 3 and is_valid_medical_term(term) and term in note_text:
                    return term
            
            # Fallback: try original value if cleaning broke it
            if value in note_text:
                return value
            # Fallback: try extracting individual words (for compound descriptions)
            words = value.split()
            for word in words:
                if len(word) > 5 and word in note_text:  # Only longer words
                    return word
    
    return None


def _extract_shortest_medical_term(phrase: str, note_text: str) -> Optional[str]:
    """Extract the shortest meaningful medical term from a compound phrase."""
    # Common patterns to extract from long phrases:
    # "severe right shoulder pain and inability..." -> "shoulder pain"
    # "7-day history of progressively worsening cough" -> "cough"
    
    # Pattern 1: Extract "X pain" or "X dysfunction"
    symptom_patterns = [
        r'(\w+\s+pain)',
        r'(\w+\s+dysfunction)',
        r'(\w+\s+failure)',
        r'(\w+\s+disease)',
        r'(\w+\s+syndrome)',
    ]
    for pattern in symptom_patterns:
        match = re.search(pattern, phrase, re.IGNORECASE)
        if match:
            term = match.group(1)
            if term in note_text:
                return term
    
    # Pattern 2: Extract last 1-2 words (often the key medical term)
    words = phrase.split()
    if len(words) >= 2:
        # Try last 2 words
        last_two = " ".join(words[-2:])
        if last_two in note_text and len(last_two) > 5:
            return last_two
        # Try last word
        last_one = words[-1]
        if last_one in note_text and len(last_one) > 5:
            return last_one
    
    return None


def extract_temporal_term(note_text: str) -> Optional[str]:
    """Extract a time expression from the note using regex."""
    patterns = [
        # X-unit history patterns
        r'\d+-(?:year|month|week|day|hour)\s+history',
        # "past/last X units" patterns
        r'(?:past|last|over the past|for the past|for)\s+\d+\s+(?:years?|months?|weeks?|days?|hours?)',
        # "X units ago" patterns
        r'\d+\s+(?:years?|months?|weeks?|days?|hours?)\s+ago',
        # "past few/several" patterns
        r'(?:past|last)\s+(?:few|several|couple of)\s+(?:years?|months?|weeks?|days?)',
        # "X-unit course/duration" patterns
        r'\d+-(?:year|month|week|day|hour)\s+(?:course|duration|period)',
        # "X years/months of" patterns
        r'\d+\s+(?:years?|months?|weeks?|days?)\s+of\s+',
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
        
        # API clients
        self.rxnorm = RxNormLookup()
        
        # No local verifier for now
        self.verifier = None
    
    # Map note type (error_type) to PRIORITIZED change types
    # First item is most appropriate for the error_type
    NOTE_TYPE_TO_CHANGE_TYPES = {
        "diagnosis": ["pseudo_factual", "temporal_rephrasing"],  # diagnosis → UMLS synonym
        "pharmacotherapy": ["equivalent_citation", "pseudo_factual"],  # drug → drug class (RxNorm)
        "treatment": ["equivalent_citation", "pseudo_factual", "temporal_rephrasing"],  # treatment drugs
        "management": ["equivalent_citation", "temporal_rephrasing", "pseudo_factual"],  # management plans
        "causalorganism": ["equivalent_citation", "pseudo_factual"],  # antibiotics → class OR organism synonym
        # fallback: all types if not specified
    }

    # Map error_type to preferred extraction fields - PRIORITIZED by relevance
    ERROR_TYPE_TO_FIELDS = {
        "diagnosis": [
            "course_and_outcome.diagnosis_primary",  # Primary target
            "history.past_medical_history",
            "history.chief_complaint",
            "examination.physical_exam_findings",  # NEW: symptom descriptions
            "history.history_of_present_illness",  # NEW: detailed symptoms
        ],
        "pharmacotherapy": [
            "course_and_outcome.plan_and_treatment",  # Where drugs are prescribed
            "course_and_outcome.hospital_course_summary",  # Also contains drug info
            "history.medications_prior_to_admission",
            "course_and_outcome.procedures_performed",  # NEW: sometimes contains medications
        ],
        "treatment": [
            "course_and_outcome.plan_and_treatment",
            "course_and_outcome.hospital_course_summary",
            "history.medications_prior_to_admission",
            "course_and_outcome.procedures_performed",  # NEW
        ],
        "management": [
            "course_and_outcome.hospital_course_summary",  # Management decisions
            "course_and_outcome.plan_and_treatment",
            "history.medications_prior_to_admission",
            "course_and_outcome.procedures_performed",  # NEW
        ],
        "causalorganism": [
            "clinical_data.lab_results",  # For equivalent_citation (antibiotics)
            "clinical_data.microbiology",  # For pseudo_factual (organism)
        ],
        # fallback: []
    }

    # ============================================
    # CASCADING FIELD PRIORITIES FOR EXTRACTION
    # ============================================
    
    # For DIAGNOSIS notes - cascading priorities
    PRIMARY_DIAGNOSIS_FIELDS = [
        "course_and_outcome.diagnosis_primary",
    ]
    SECONDARY_DIAGNOSIS_FIELDS = [
        "history.past_medical_history",
        "history.chief_complaint",
    ]
    ALL_DIAGNOSIS_FIELDS = [
        "course_and_outcome.diagnosis_primary",
        "history.past_medical_history",
        "history.chief_complaint",
        "examination.physical_exam_findings",
        "history.history_of_present_illness",
        "clinical_data.lab_results",
        "clinical_data.imaging_results",
    ]
    
    # For MEDICATION notes - cascading priorities  
    PRIMARY_MEDICATION_FIELDS = [
        "history.medications_prior_to_admission",
        "course_and_outcome.plan_and_treatment",
    ]
    SECONDARY_MEDICATION_FIELDS = [
        "course_and_outcome.hospital_course_summary",
        "course_and_outcome.procedures_performed",
    ]
    ALL_MEDICATION_FIELDS = [
        "history.medications_prior_to_admission",
        "course_and_outcome.plan_and_treatment",
        "course_and_outcome.hospital_course_summary",
        "course_and_outcome.procedures_performed",
        "history.past_medical_history",
    ]

    def get_allowed_change_types(self, note_type: str) -> list:
        """Return allowed change types for a given note type."""
        return self.NOTE_TYPE_TO_CHANGE_TYPES.get(note_type, self.change_types)
    
    def _verify_umls_has_good_synonym(self, term: str) -> bool:
        """Check if UMLS has a good quality synonym for this term (without selecting one)."""
        result = get_pseudo_factual_replacement(term)
        return result.verified
    
    def _try_extract_with_verification(self, extraction: dict, fields: list, note_text: str, change_type: str) -> Optional[str]:
        """
        Extract a term from fields and verify UMLS has a good synonym for it.
        Returns the term only if verification succeeds.
        """
        for field in fields:
            value = get_value_from_path(extraction, field)
            if not value:
                continue
            
            terms = []
            if isinstance(value, list):
                terms = [t for t in value if t]
            elif isinstance(value, str):
                # Split on commas for compound values
                terms = [t.strip() for t in value.split(",") if t.strip()]
            
            for term in terms:
                # Clean the term
                cleaned = clean_extracted_term(term)
                if not cleaned or not is_valid_medical_term(cleaned):
                    continue
                
                # Must exist in note text
                if cleaned not in note_text:
                    # Try original if cleaning broke it
                    if term in note_text:
                        cleaned = term
                    else:
                        continue
                
                # For pseudo_factual, verify UMLS has a good synonym
                if change_type == "pseudo_factual":
                    if self._verify_umls_has_good_synonym(cleaned):
                        return cleaned
                else:
                    # For other change types, just return if valid
                    return cleaned
        
        return None
    
    def _extract_medical_terms_from_text(self, note_text: str, change_type: str) -> Optional[str]:
        """
        Last resort: scan the note text directly for medical terms.
        Uses regex patterns to find potential medical terms.
        """
        # Common medical term patterns
        patterns = [
            # Disease/condition patterns
            r'\b([A-Z][a-z]+ (?:disease|syndrome|disorder|deficiency))\b',
            r'\b((?:acute|chronic) [a-z]+ [a-z]+)\b',
            # Drug patterns (capitalized single words followed by context)
            r'(?:started on|prescribed|given|taking) ([A-Za-z]+)\b',
            # Diagnosis patterns
            r'diagnosed with ([A-Za-z]+(?: [a-z]+)?)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, note_text, re.IGNORECASE)
            for match in matches:
                term = match.strip()
                if len(term) > 4 and is_valid_medical_term(term):
                    if change_type == "pseudo_factual":
                        if self._verify_umls_has_good_synonym(term):
                            return term
                    else:
                        return term
        
        return None
    
    def get_verified_replacement(self, term: str, change_type: str) -> Optional[dict]:
        """Get a verified replacement using the unified medical_knowledge_base API."""
        
        if change_type == "pseudo_factual":
            result = get_pseudo_factual_replacement_with_fallback(term)
            if result.verified:
                return {
                    "original": term,
                    "replacement": result.replacement,
                    "source": result.source,
                    "verified": True
                }
        
        elif change_type == "temporal_rephrasing":
            result = get_temporal_replacement(term)
            if result.verified:
                return {
                    "original": term,
                    "replacement": result.replacement,
                    "source": result.source,
                    "verified": True
                }
        
        elif change_type == "equivalent_citation":
            # Try to extract just the drug name if term contains extra words
            drug_name = self._extract_drug_name(term)
            result = get_equivalent_citation_replacement(drug_name)
            if result.verified:
                return {
                    "original": drug_name,
                    "replacement": result.replacement,
                    "source": result.source,
                    "verified": True
                }
        
        return None
    
    def _extract_drug_name(self, term: str) -> str:
        """Extract just the drug name from a medication string like 'albuterol inhaler as needed'."""
        # Common suffixes/phrases to strip
        strip_patterns = [
            r'\s+(?:inhaler|tablet|capsule|injection|solution|cream|ointment|patch|spray)s?.*$',
            r'\s+(?:as needed|prn|daily|twice daily|bid|tid|qid|qhs|qd).*$',
            r'\s+\d+\s*(?:mg|mcg|ml|g|units?).*$',
            r'\s+(?:oral|iv|im|sq|topical|sublingual).*$',
        ]
        
        result = term.strip()
        for pattern in strip_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # If we stripped everything, return first word as drug name
        if not result.strip():
            result = term.split()[0] if term.split() else term
        
        return result.strip()
    
    # Non-drug terms that RxNorm can't handle
    NON_DRUG_TERMS = [
        # Procedures
        "infusion", "injection", "therapy", "treatment", "procedure",
        "surgery", "operation", "transplant", "dialysis", "transfusion",
        "vaccination", "immunization", "screening", "test", "exam",
        "monitoring", "observation", "consultation", "referral",
        # Medical equipment/supplies
        "tube", "catheter", "line", "drain", "device", "pump", "monitor",
        "nasogastric", "endotracheal", "foley", "picc", "central line",
        # Delivery methods
        "oral", "intravenous", "subcutaneous", "intramuscular", "topical",
        "inhaler", "nebulizer", "patch", "spray", "suppository",
        # Substances/solutions
        "saline", "solution", "fluid", "water", "glucose", "dextrose",
        "sodium", "potassium", "electrolyte", "oxygen", "air",
        "antacid", "supplement", "vitamin",
        # Generic terms
        "medication", "drug", "agent", "compound", "preparation",
    ]
    
    def _is_likely_drug(self, term: str) -> bool:
        """Check if term is likely a drug name (not a procedure/delivery method)."""
        term_lower = term.lower().strip()
        
        # Filter out non-drug terms
        for non_drug in self.NON_DRUG_TERMS:
            if non_drug in term_lower:
                return False
        
        # Must be a reasonable length for drug name
        if len(term_lower) < 4 or len(term_lower) > 30:
            return False
        
        return True
    
    # Common antibiotics for extraction from lab results
    COMMON_ANTIBIOTICS = [
        "Penicillin G", "Penicillin", "Ampicillin", "Amoxicillin", "Piperacillin",
        "Cefazolin", "Ceftriaxone", "Cefepime", "Ceftazidime", "Cephalexin",
        "Azithromycin", "Erythromycin", "Clarithromycin",
        "Doxycycline", "Tetracycline", "Minocycline",
        "Ciprofloxacin", "Levofloxacin", "Moxifloxacin",
        "Vancomycin", "Linezolid", "Daptomycin",
        "Imipenem", "Meropenem", "Ertapenem",
        "Gentamicin", "Tobramycin", "Amikacin",
        "Metronidazole", "Clindamycin", "Trimethoprim", "Sulfamethoxazole",
        "Nitrofurantoin", "Fosfomycin", "Rifampin",
    ]
    
    def _extract_antibiotic_from_lab(self, lab_results: str, note_text: str) -> Optional[str]:
        """Extract an antibiotic name from lab results (e.g., susceptibility testing)."""
        if not lab_results:
            return None
        
        # Try to find antibiotics in the lab results that also appear in the note
        for antibiotic in self.COMMON_ANTIBIOTICS:
            if antibiotic in lab_results and antibiotic in note_text:
                return antibiotic
        
        # Also try case-insensitive matching
        for antibiotic in self.COMMON_ANTIBIOTICS:
            pattern = re.compile(re.escape(antibiotic), re.IGNORECASE)
            if pattern.search(lab_results) and pattern.search(note_text):
                match = pattern.search(note_text)
                return match.group(0)  # Return as it appears in note
        
        return None
    
    def find_target_term(self, note_text: str, extraction: dict, change_type: str, note_type: str = None) -> Optional[str]:
        """
        Find a suitable target term based on change type, extraction, and note type.
        Uses CASCADING EXTRACTION with UMLS verification for pseudo_factual:
        1. Try PRIMARY fields first, verify UMLS has good synonym
        2. If not found, try SECONDARY fields with verification
        3. If still not found, try ALL fields with verification
        4. Last resort: scan note text directly
        """
        # Determine note_type (error_type)
        if note_type is None:
            note_type = extraction.get("error_type") or extraction.get("note_type") or ""

        config = self.configs[change_type]

        # For temporal_rephrasing, use regex
        if change_type == "temporal_rephrasing":
            return extract_temporal_term(note_text)

        # For irrelevant_correlation, use regex
        if change_type == "irrelevant_correlation":
            term = extract_irrelevant_term(note_text)
            if term:
                return term
        
        # For equivalent_citation, look in medication fields AND lab_results (for antibiotics)
        if change_type == "equivalent_citation":
            # Use cascading for medications
            term = self._try_extract_with_verification(
                extraction, self.PRIMARY_MEDICATION_FIELDS, note_text, change_type
            )
            if term:
                drug_name = self._extract_drug_name(term)
                if self._is_likely_drug(drug_name):
                    return drug_name
            
            # Secondary medication fields
            term = self._try_extract_with_verification(
                extraction, self.SECONDARY_MEDICATION_FIELDS, note_text, change_type
            )
            if term:
                drug_name = self._extract_drug_name(term)
                if self._is_likely_drug(drug_name):
                    return drug_name
            
            # Fallback: extract antibiotics from lab_results (for causalorganism notes)
            lab_results = get_value_from_path(extraction, "clinical_data.lab_results")
            if lab_results:
                antibiotic = self._extract_antibiotic_from_lab(lab_results, note_text)
                if antibiotic:
                    return antibiotic
            return None

        # For pseudo_factual, use CASCADING EXTRACTION WITH VERIFICATION
        if change_type == "pseudo_factual":
            # STAGE 1: Try PRIMARY diagnosis fields with UMLS verification
            term = self._try_extract_with_verification(
                extraction, self.PRIMARY_DIAGNOSIS_FIELDS, note_text, change_type
            )
            if term:
                return term
            
            # STAGE 2: Try SECONDARY diagnosis fields with verification
            term = self._try_extract_with_verification(
                extraction, self.SECONDARY_DIAGNOSIS_FIELDS, note_text, change_type
            )
            if term:
                return term
            
            # STAGE 3: Try ALL diagnosis fields with verification
            term = self._try_extract_with_verification(
                extraction, self.ALL_DIAGNOSIS_FIELDS, note_text, change_type
            )
            if term:
                return term
            
            # STAGE 4: Try corrected_sentence
            corrected = extraction.get("corrected_sentence", "")
            if corrected:
                term = self._extract_term_from_corrected_sentence(corrected, note_text)
                if term and self._verify_umls_has_good_synonym(term):
                    return term
            
            # STAGE 5: Last resort - scan note text directly
            term = self._extract_medical_terms_from_text(note_text, change_type)
            if term:
                return term
            
            return None

        # For other change types, use the old preferred fields approach
        preferred_fields = self.ERROR_TYPE_TO_FIELDS.get(note_type, [])
        target_sections = config.get("target_sections", [])

        # Try preferred fields first
        if preferred_fields:
            term = extract_target_term(extraction, preferred_fields, note_text)
            if term:
                return term

        # Fallback to config target_sections
        term = extract_target_term(extraction, target_sections, note_text)
        if term:
            return term
        
        # Fallback: try corrected_sentence for all types
        corrected = extraction.get("corrected_sentence", "")
        if corrected:
            term = self._extract_term_from_corrected_sentence(corrected, note_text)
            if term:
                return term
        
        return None
    
    def _extract_term_from_corrected_sentence(self, corrected: str, note_text: str) -> Optional[str]:
        """Extract a medical term from the corrected_sentence field."""
        # Common patterns to look for in corrected sentences
        # e.g., "Suspected of Huntington disease." -> "Huntington disease"
        # e.g., "Treatment with ranibizumab is initiated." -> "ranibizumab"
        
        # Pattern 1: "Treatment with X" or "started on X"
        drug_patterns = [
            r'(?:treatment with|started on|initiated on|prescribed)\s+([A-Za-z]+)',
            r'([A-Za-z]+)\s+(?:is initiated|is started|is prescribed)',
        ]
        for pattern in drug_patterns:
            match = re.search(pattern, corrected, re.IGNORECASE)
            if match:
                term = match.group(1)
                # Clean and validate
                term = clean_extracted_term(term)
                if term and is_valid_medical_term(term) and len(term) > 2 and term in note_text:
                    return term
        
        # Pattern 2: "diagnosed with X" or "diagnosis of X"
        diagnosis_patterns = [
            r'(?:diagnosed with|diagnosis of|suspected of)\s+([A-Za-z\s]+?)(?:\.|$|,)',
            r'([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:is the organism|was identified)',
        ]
        for pattern in diagnosis_patterns:
            match = re.search(pattern, corrected, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                # Clean and validate
                term = clean_extracted_term(term)
                if len(term) > 3 and is_valid_medical_term(term) and term in note_text:
                    return term
        
        return None
    
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
        
        # If no target term found and not logical_restatement, fail early or try fallback
        if not target_term and change_type != "logical_restatement":
            error_msg = f"No target term found for change_type '{change_type}'"
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
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}")
            print(f"[DEBUG] Raw LLM response: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
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
                "error": f"JSON parsing error: {str(e)}"
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