"""
Medical Knowledge Base v2 - Clean, Modular Design

Supports 4 verified transformation types:
1. Pseudo-Factual Substitution: Medical term → equivalent term (UMLS CUI matching)
2. Temporal Rephrasing: Time expressions (Deterministic rules)
3. Irrelevant Correlation: Non-clinical details (Curated lists)
4. Equivalent Citation: Drug → class (RxNorm), Radiology terms (RadLex)
"""

import requests
import re
import os
import json
import hashlib
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env')
except ImportError:
    pass


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class VerifiedReplacement:
    """Result of a verified term replacement."""
    original: str
    replacement: str
    verified: bool
    source: str  # 'UMLS', 'RxNorm', 'temporal', 'curated'
    cui: Optional[str] = None
    semantic_type: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "replacement": self.replacement if self.verified else None,
            "selected_synonym": self.replacement if self.verified else None,
            "verified": self.verified,
            "source": self.source,
            "cui": self.cui,
            "semantic_type": self.semantic_type,
        }


# ============================================
# SIMPLE CACHE
# ============================================

_CACHE: Dict[str, any] = {}

def _cache_key(prefix: str, **kwargs) -> str:
    data = json.dumps(kwargs, sort_keys=True)
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"

def _cached(key: str) -> Optional[any]:
    return _CACHE.get(key)

def _cache(key: str, value: any) -> any:
    _CACHE[key] = value
    return value


# ============================================
# 1. PSEUDO-FACTUAL SUBSTITUTION (UMLS)
# ============================================

class UMLSLookup:
    """
    Simple UMLS lookup for pseudo-factual substitutions.
    Returns synonyms from the SAME CUI only (guaranteed semantic equivalence).
    """
    
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    # Expanded sources for quality and diversity
    # CHV = Consumer Health Vocabulary (lay/patient-friendly terms)
    # MDR = MedDRA (clinical variations)
    TRUSTED_SOURCES = ["SNOMEDCT_US", "NCI", "MSH", "MTH", "CHV", "MDR"]
    
    # Expanded term types for more diverse synonyms
    # PT = Preferred Term, SY = Synonym
    # ET = Entry Term (informal names), LLT = Lower Level Term (MedDRA variations)
    TRUSTED_TTYS = ["PT", "SY", "ET", "LLT"]
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UMLS_API_KEY")
        if not self.api_key:
            raise ValueError("UMLS_API_KEY required")
        self.session = requests.Session()
    
    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make cached API request."""
        params = params or {}
        params["apiKey"] = self.api_key
        
        key = _cache_key("umls", url=url, params=params)
        if (cached := _cached(key)) is not None:
            return cached
        
        try:
            resp = self.session.get(url, params=params, timeout=15)
            if resp.ok:
                return _cache(key, resp.json())
        except Exception:
            pass
        return None
    
    def get_cui(self, term: str) -> Optional[str]:
        """Get CUI for a term."""
        url = f"{self.BASE_URL}/search/current"
        params = {
            "string": term,
            "searchType": "exact",
            "sabs": ",".join(self.TRUSTED_SOURCES),
            "pageSize": 5,
        }
        data = self._get(url, params)
        if data:
            results = data.get("result", {}).get("results", [])
            if results:
                return results[0].get("ui")
        return None
    
    def get_synonyms_for_cui(self, cui: str) -> List[dict]:
        """Get all synonyms for a CUI with metadata (same concept = guaranteed equivalence).
        
        Returns list of dicts with keys: name, source, tty
        """
        url = f"{self.BASE_URL}/content/current/CUI/{cui}/atoms"
        params = {
            "language": "ENG",
            "sabs": ",".join(self.TRUSTED_SOURCES),
            "ttys": ",".join(self.TRUSTED_TTYS),
            "pageSize": 100,  # Increased for more options
        }
        data = self._get(url, params)
        if not data:
            return []
        
        atoms = data.get("result", [])
        synonyms = []
        seen = set()
        
        for atom in atoms:
            name = atom.get("name", "").strip()
            name_lower = name.lower()
            source = atom.get("rootSource", "")
            tty = atom.get("termType", "")
            
            # Skip codes, duplicates
            if not name or name_lower in seen:
                continue
            if re.match(r'^[A-Z0-9\-\.]+$', name):
                continue
            
            seen.add(name_lower)
            synonyms.append({"name": name, "source": source, "tty": tty})
        
        return synonyms
    
    def get_semantic_type(self, cui: str) -> Optional[str]:
        """Get primary semantic type for a CUI."""
        url = f"{self.BASE_URL}/content/current/CUI/{cui}"
        data = self._get(url)
        if data and "result" in data:
            types = data["result"].get("semanticTypes", [])
            if types:
                return types[0].get("name")
        return None


# ============================================
# TERM DECOMPOSITION HELPERS
# ============================================

# Common modifiers that can be stripped to find core medical term
POSITIONAL_MODIFIERS = {
    'upper', 'lower', 'left', 'right', 'anterior', 'posterior', 
    'lateral', 'medial', 'proximal', 'distal', 'central', 'peripheral',
    'superior', 'inferior', 'ventral', 'dorsal', 'bilateral', 'unilateral',
}

DESCRIPTIVE_MODIFIERS = {
    'acute', 'chronic', 'mild', 'moderate', 'severe', 'primary', 'secondary',
    'early', 'late', 'advanced', 'progressive', 'recurrent', 'persistent',
    'localized', 'generalized', 'focal', 'diffuse', 'partial', 'complete',
    'simple', 'complex', 'benign', 'malignant',
}

IMAGING_PREFIXES = {
    'ct', 'mri', 'mr', 'pet', 'spect', 'x-ray', 'xray', 'ultrasound', 'us',
    'doppler', 'contrast', 'non-contrast', 'enhanced', 'guided',
}

ALL_MODIFIERS = POSITIONAL_MODIFIERS | DESCRIPTIVE_MODIFIERS | IMAGING_PREFIXES

# Medical abbreviation expansions for term decomposition
ABBREVIATION_EXPANSIONS = {
    'mri': 'Magnetic resonance imaging',
    'ct': 'Computed tomography',
    'pet': 'Positron emission tomography',
    'spect': 'Single photon emission computed tomography',
    'ssri': 'Selective serotonin reuptake inhibitor',
    'snri': 'Serotonin norepinephrine reuptake inhibitor',
    'maoi': 'Monoamine oxidase inhibitor',
    'nsaid': 'Nonsteroidal anti-inflammatory drug',
    'copd': 'Chronic obstructive pulmonary disease',
    'gerd': 'Gastroesophageal reflux disease',
    'cad': 'Coronary artery disease',
    'chf': 'Congestive heart failure',
    'afib': 'Atrial fibrillation',
    'dvt': 'Deep vein thrombosis',
    'pe': 'Pulmonary embolism',
    'mi': 'Myocardial infarction',
    'cvd': 'Cardiovascular disease',
    'uti': 'Urinary tract infection',
}

# US/UK spelling normalization
UK_TO_US_SPELLING = {
    'behaviour': 'behavior', 'colour': 'color', 'favour': 'favor',
    'honour': 'honor', 'labour': 'labor', 'tumour': 'tumor',
    'oedema': 'edema', 'anaemia': 'anemia', 'haemorrhage': 'hemorrhage',
    'diarrhoea': 'diarrhea', 'oesophagus': 'esophagus', 'paediatric': 'pediatric',
    'foetus': 'fetus', 'gynaecology': 'gynecology', 'leukaemia': 'leukemia',
}

# Synonym quality filters (consolidated patterns)
FILTER_PATTERNS = {
    'product_suffixes': ['-containing', '-based', ' product', ' preparation',
                         ' formulation', ' tablet', ' capsule', ' injection',
                         ' solution', ' suspension'],
    'archaic_terms': ['praecox', 'morbus', 'febris', 'hydrops', 'phthisis'],
    'misspellings': ['diabete ', 'diabete mellitus', 'sclerosi', 'hypertensio ', 'hypotensio '],
    'awkward_endings': ['high', 'low', 'negative', 'positive', 'acute', 'chronic',
                        'generalized', 'localized', 'severe', 'mild'],
    'common_adjectives': {'high', 'low', 'acute', 'chronic', 'severe', 'mild',
                          'primary', 'secondary', 'normal', 'abnormal'},
    'noun_suffixes': ['ness', 'tion', 'sion', 'ment', 'ity', 'ance', 'ence'],
    'adjective_suffixes': ['ble', 'ive', 'ous', 'ful', 'less', 'ing', 'ed', 'al', 'ic'],
    'common_words': {'mobile', 'stable', 'fixed', 'clear', 'normal', 'regular'},
    'technical_terms': ['mri', 'ct', 'scan', 'r1', 'r2', 'lipid', 't1', 't2'],
    'plural_organs': ['colons', 'kidneys', 'lungs', 'hearts', 'livers', 'intestines', 'bowels'],
    'followup_adjectives': ['spastic', 'acute', 'chronic', 'enlarged', 'inflamed', 'irritable'],
}

BAD_GRAMMAR_PATTERNS = [
    r'^attempts\s', r'^increases?\s', r'^decreases?\s', r'^feeling\s',
    r'\ssuicides$', r'^disorders?\s', r'\ssymptom$', r'^off\s', r'off food',
    r'^on\s', r'^losing\s', r'losing wt', r'^gaining\s', r'\swt$', r'\swt\s',
]


def _normalize_spelling(words: set) -> set:
    """Normalize common US/UK spelling variants."""
    return {UK_TO_US_SPELLING.get(w, w) for w in words}


def extract_core_term(term: str) -> Optional[str]:
    """
    Extract the core medical term by stripping common modifiers.
    
    Examples:
        "upper endoscopy" → "endoscopy"
        "CT angiogram" → "angiogram"
        "acute cholangitis" → "cholangitis"
        "left knee pain" → "knee pain"
    
    Returns None if no simplification possible.
    """
    term_lower = term.lower().strip()
    words = term_lower.split()
    
    if len(words) <= 1:
        return None
    
    # Try removing leading modifiers
    core_words = []
    found_core = False
    for word in words:
        if word in ALL_MODIFIERS and not found_core:
            continue  # Skip leading modifier
        else:
            found_core = True
            core_words.append(word)
    
    if core_words and len(core_words) < len(words):
        return ' '.join(core_words)
    
    # Try removing trailing modifiers (less common but possible)
    # e.g., "endoscopy upper" (rare)
    
    return None


def get_term_variants(term: str) -> List[str]:
    """
    Get variants of a term to try for lookup.
    
    Returns list in order of preference:
    1. Original term
    2. Expanded abbreviations (MRI → Magnetic resonance imaging)
    3. Core term (modifiers stripped)
    4. Imaging modality (MRI, CT, etc.) if present
    5. Last significant word (for compound terms)
    """
    variants = [term]
    words = term.split()
    term_lower = term.lower()
    
    # Try expanding abbreviations in the term
    # e.g., "cranial MRI with contrast" → "cranial Magnetic resonance imaging with contrast"
    expanded_term = term
    for word in words:
        if word.lower() in ABBREVIATION_EXPANSIONS:
            expanded = ABBREVIATION_EXPANSIONS[word.lower()]
            expanded_term = expanded_term.replace(word, expanded)
    if expanded_term != term:
        variants.append(expanded_term)
    
    # Also try just the expanded abbreviation alone if term is single abbreviation
    if len(words) == 1 and term_lower in ABBREVIATION_EXPANSIONS:
        pass  # Already added above
    # Or if term contains abbreviation, try the expansion alone
    elif any(word.lower() in ABBREVIATION_EXPANSIONS for word in words):
        for word in words:
            if word.lower() in ABBREVIATION_EXPANSIONS:
                exp = ABBREVIATION_EXPANSIONS[word.lower()]
                if exp not in variants and exp.lower() not in [v.lower() for v in variants]:
                    variants.append(exp)
    
    core = extract_core_term(term)
    if core and core != term_lower:
        variants.append(core)
    
    # For imaging terms, extract the modality (MRI, CT, etc.) as a valid variant
    # e.g., "cranial MRI with contrast" → "MRI"
    IMAGING_MODALITIES = {'mri', 'ct', 'pet', 'spect', 'ultrasound', 'x-ray', 'xray'}
    for word in words:
        if word.lower() in IMAGING_MODALITIES:
            if word not in variants and word.lower() not in [v.lower() for v in variants]:
                variants.append(word.upper() if word.upper() in ['MRI', 'CT', 'PET'] else word)
    
    # For "X Y with Z" patterns, try "Y" (e.g., "cranial MRI with contrast" → already handled above)
    # Also try the word before "with" if present
    if 'with' in term_lower:
        parts = term_lower.split('with')[0].strip().split()
        if len(parts) >= 1:
            last_before_with = parts[-1]
            if last_before_with not in [v.lower() for v in variants] and len(last_before_with) > 2:
                variants.append(last_before_with)
    
    # For multi-word terms, try the last word (often the main concept)
    if len(words) >= 2:
        last_word = words[-1]
        if last_word.lower() not in ALL_MODIFIERS and len(last_word) > 3:
            if last_word not in variants and last_word.lower() not in [v.lower() for v in variants]:
                variants.append(last_word)
    
    return variants


# Clinical semantic types that are valid for pseudo-factual substitution
# These are medical/clinical concepts where synonym substitution makes sense
CLINICAL_SEMANTIC_TYPES = {
    # Disorders
    "Disease or Syndrome",
    "Neoplastic Process",
    "Mental or Behavioral Dysfunction",
    "Cell or Molecular Dysfunction",
    "Congenital Abnormality",
    "Acquired Abnormality",
    "Injury or Poisoning",
    "Pathologic Function",
    # Findings & Symptoms
    "Sign or Symptom",
    "Finding",
    "Laboratory or Test Result",
    # Anatomy
    "Body Part, Organ, or Organ Component",
    "Body Location or Region",
    "Body System",
    "Tissue",
    "Cell",
    # Chemicals & Drugs
    "Pharmacologic Substance",
    "Clinical Drug",
    "Antibiotic",
    "Organic Chemical",
    "Amino Acid, Peptide, or Protein",
    "Hormone",
    "Enzyme",
    "Vitamin",
    "Immunologic Factor",
    # Procedures
    "Therapeutic or Preventive Procedure",
    "Diagnostic Procedure",
    "Laboratory Procedure",
    "Health Care Activity",
    # Organisms
    "Bacterium",
    "Virus",
    "Fungus",
    "Organism",
    # Other medical
    "Medical Device",
    "Biomedical or Dental Material",
    "Clinical Attribute",
    "Biologic Function",
    "Physiologic Function",
    "Organ or Tissue Function",
}

# Non-clinical semantic types to explicitly reject
# These are generic concepts where substitution produces poor results
NON_CLINICAL_SEMANTIC_TYPES = {
    "Qualitative Concept",      # "deep", "high", "low", "severe"
    "Quantitative Concept",     # numbers, amounts
    "Spatial Concept",          # "anterior", "proximal"
    "Temporal Concept",         # time-related
    "Functional Concept",       # abstract functions
    "Idea or Concept",          # abstract ideas
    "Intellectual Product",     # documents, guidelines
    "Language",                 # linguistic terms
    "Occupation or Discipline", # "physician", "nurse"
    "Organization",             # hospitals, companies
    "Professional or Occupational Group",
    "Population Group",
    "Age Group",
    "Patient or Disabled Group",
    "Family Group",
    "Group",
    "Geographic Area",
    "Activity",                 # generic activities
    "Daily or Recreational Activity",
    "Governmental or Regulatory Activity",
    "Educational Activity",
    "Machine Activity",
    "Human-caused Phenomenon or Process",
    "Natural Phenomenon or Process",
}


def get_pseudo_factual_replacement(term: str) -> VerifiedReplacement:
    """
    Get a verified pseudo-factual replacement.
    
    Strategy: Find CUI for term, get synonyms from SAME CUI.
    This guarantees semantic equivalence (same concept, different name).
    
    Examples:
        "hypertension" → "high blood pressure" (same CUI: C0020538)
        "myocardial infarction" → "heart attack" (same CUI)
    """
    term_clean = term.strip()
    term_lower = term_clean.lower()
    
    # FILTER 0: Reject very short terms/abbreviations (too ambiguous)
    # e.g., "CT" can mean computed tomography, calcitonin, etc.
    # Single words with ≤3 characters are too risky
    if len(term_clean.split()) == 1 and len(term_clean) <= 3:
        return VerifiedReplacement(term, "", False, "UMLS")
    
    # Check cache
    key = _cache_key("pseudo_factual", term=term_lower)
    if (cached := _cached(key)) is not None:
        return cached
    
    try:
        umls = UMLSLookup()
        
        # Get CUI for original term
        cui = umls.get_cui(term_clean)
        if not cui:
            return _cache(key, VerifiedReplacement(term, "", False, "UMLS"))
        
        # Check semantic type - reject non-clinical concepts
        sem_type = umls.get_semantic_type(cui)
        if sem_type:
            # Reject if explicitly non-clinical
            if sem_type in NON_CLINICAL_SEMANTIC_TYPES:
                return _cache(key, VerifiedReplacement(term, "", False, "UMLS", cui, sem_type))
            # If not in our clinical list and not explicitly rejected, be cautious
            if sem_type not in CLINICAL_SEMANTIC_TYPES:
                # Allow only if it's clearly medical-sounding
                pass  # We'll let it through but other filters may catch it
        
        # Get synonyms from same CUI (now returns list of dicts with metadata)
        synonyms = umls.get_synonyms_for_cui(cui)
        
        # Score and filter candidates
        scored_candidates = []
        term_words = set(term_lower.split())
        
        for syn in synonyms:
            name = syn["name"]
            source = syn["source"]
            tty = syn["tty"]
            name_lower = name.lower()
            name_words = set(name_lower.split())
            
            # ==========================================
            # HARD FILTERS (reject outright)
            # ==========================================
            
            # Basic identity check
            if name_lower == term_lower:
                continue
            if len(name) < 3 or len(name) > 60:
                continue
            if name_lower.endswith(term_lower):  # No "X syndrome" for "X"
                continue
            
            # FILTER 1: Reject ICD-style qualifiers (commas, parentheses)
            # e.g., "osteomalacia, unspecified", "depression (finding)"
            if ',' in name or '(' in name or ')' in name:
                continue
            
            # FILTER 2: Reject word-order swaps (same words rearranged)
            # e.g., "watery diarrhea" → "diarrhea watery", "allergic rhinitis" → "rhinitis allergic"
            # Also catches "Langerhans cell histiocytosis" → "Cell granulomatosis langerhans" (high overlap)
            if term_words == name_words:
                continue
            
            # Check word overlap ratio - reject if >60% same words (likely just rearrangement)
            # Lowered from 80% to catch more word-order swaps
            if len(term_words) > 1 and len(name_words) > 1:
                overlap = len(term_words & name_words)
                overlap_ratio = overlap / max(len(term_words), len(name_words))
                if overlap_ratio >= 0.6:
                    continue
            
            # FILTER 2b: Reject awkward word orders (noun before adjective patterns)
            if name_words and list(name_lower.split())[-1] in FILTER_PATTERNS['awkward_endings']:
                if len(name_words) >= 2:
                    continue
            
            # Check with US/UK spelling normalization
            if _normalize_spelling(term_words) == _normalize_spelling(name_words):
                continue
            
            # FILTER 3: Reject definitions/explanations containing "or", action phrases
            # e.g., "push down or depress", "to make less intense"
            if ' or ' in name_lower:
                continue
            if name_lower.startswith(('to ', 'a ', 'the ')):
                continue
            
            # FILTER 4: Reject product/formulation names
            if any(p in name_lower for p in FILTER_PATTERNS['product_suffixes']):
                continue
            
            # FILTER 5: Reject ICD "NOS" (Not Otherwise Specified) qualifiers
            # e.g., "cholangitis acute nos", "hypertension nos"
            if ' nos' in name_lower or name_lower.endswith(' nos'):
                continue
            
            # FILTER 6: Reject archaic/obsolete medical terms
            if any(a in name_lower for a in FILTER_PATTERNS['archaic_terms']):
                continue
            
            # FILTER 7: Reject grammatically broken forms
            if any(re.search(p, name_lower) for p in BAD_GRAMMAR_PATTERNS):
                continue
            
            # FILTER 7d: Reject obvious misspellings/typos in UMLS
            if any(typo in name_lower for typo in FILTER_PATTERNS['misspellings']):
                continue
            
            # FILTER 7b: Reject adjective→noun mismatches
            term_is_adj = any(term_lower.endswith(s) for s in FILTER_PATTERNS['adjective_suffixes']) or len(term_lower.split()) == 1
            name_is_noun = any(name_lower.endswith(s) for s in FILTER_PATTERNS['noun_suffixes'])
            if term_is_adj and name_is_noun and len(term_lower.split()) == 1 and len(name_lower.split()) == 1:
                continue
            
            # FILTER 7e: Reject semantic mismatches (common word → technical jargon)
            if term_lower in FILTER_PATTERNS['common_words'] and any(t in name_lower for t in FILTER_PATTERNS['technical_terms']):
                continue
            
            # FILTER 7f: Reject plural/singular noun mismatches in compound terms
            should_skip_7f = False
            if len(name_words) >= 2:
                words_list = name_lower.split()
                # Reject "syndrome/disorder" at start
                if words_list[0] in ['syndrome', 'syndromes', 'disorder', 'disorders'] and len(words_list) > 2:
                    should_skip_7f = True
                # Reject plural nouns followed by adjectives
                if not should_skip_7f:
                    for i, word in enumerate(words_list[:-1]):
                        is_plural = word in FILTER_PATTERNS['plural_organs'] or (len(word) > 3 and word.endswith('s') and word not in ['status', 'virus', 'us'])
                        if is_plural and words_list[i + 1] in FILTER_PATTERNS['followup_adjectives']:
                            should_skip_7f = True
                            break
            if should_skip_7f:
                continue
            
            # FILTER 7c: Reject awkward word orders (noun + adjective + noun)
            name_words_list = name_lower.split()
            if len(name_words_list) == 3 and name_words_list[1] in FILTER_PATTERNS['common_adjectives']:
                continue
            
            # FILTER 8: Reject very short nonsense abbreviations (2-3 chars)
            # But allow legitimate medical abbreviations like NIDDM, COPD, GERD
            if len(name) <= 3 and name.isalpha():
                continue
            
            # FILTER 9: Calculate string similarity (reject if too similar - spelling variants)
            char_similarity = SequenceMatcher(None, term_lower, name_lower).ratio()
            if char_similarity > 0.90:  # 90% threshold - reject near-identical strings
                continue
            
            # FILTER 10: Check word-stem overlap (reject if same stem, e.g., anxiety→anxiousness)
            term_stem = term_lower.rstrip('syed')[:4] if len(term_lower) > 4 else term_lower
            name_stem = name_lower.rstrip('syed')[:4] if len(name_lower) > 4 else name_lower
            if len(term_lower.split()) == 1 and len(name_lower.split()) == 1:
                if term_stem == name_stem:
                    continue
            
            # ==========================================
            # SCORING (higher = better)
            # ==========================================
            score = 0.0
            
            # Reward lower character similarity (more different = better)
            score += (1.0 - char_similarity) * 5.0
            
            # Reward multi-word alternatives for single-word originals
            if len(term_clean.split()) == 1 and len(name.split()) > 1:
                score += 3.0
            
            # Reward Consumer Health Vocabulary (patient-friendly terms)
            if source == "CHV":
                score += 2.0
            
            # Reward Entry Terms (often more varied phrasings)
            if tty in ["ET", "LLT"]:
                score += 1.5
            
            # Penalize if original is substring of candidate or vice versa
            if term_lower in name_lower or name_lower in term_lower:
                score -= 2.0
            
            # Penalize very long clinical names
            if len(name) > 40:
                score -= 1.0
            
            scored_candidates.append((name, score, source, tty))
        
        if not scored_candidates:
            return _cache(key, VerifiedReplacement(term, "", False, "UMLS", cui))
        
        # Sort by score (highest first) and select best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = scored_candidates[0][0]
        
        # Match case
        if term_clean.islower():
            selected = selected.lower()
        elif term_clean[0].isupper() and term_clean[1:].islower():
            selected = selected.capitalize()
        
        # sem_type already retrieved above for filtering
        
        return _cache(key, VerifiedReplacement(
            original=term,
            replacement=selected,
            verified=True,
            source="UMLS",
            cui=cui,
            semantic_type=sem_type
        ))
        
    except Exception:
        return _cache(key, VerifiedReplacement(term, "", False, "UMLS"))


def get_pseudo_factual_replacement_with_fallback(term: str) -> VerifiedReplacement:
    """
    Get a verified pseudo-factual replacement with term decomposition fallback.
    
    If the full term fails, tries:
    1. Core term (with modifiers stripped)
    2. Last significant word
    
    Examples:
        "upper endoscopy" → try "endoscopy" if full term fails
        "CT angiogram" → try "angiogram" if full term fails
        "acute cholangitis" → try "cholangitis" if full term fails
    """
    # First try the full term
    result = get_pseudo_factual_replacement(term)
    if result.verified:
        return result
    
    # Try term variants (core term, last word)
    variants = get_term_variants(term)
    
    for variant in variants[1:]:  # Skip first (original term already tried)
        variant_result = get_pseudo_factual_replacement(variant)
        if variant_result.verified:
            # We found a replacement for the core term
            # Need to reconstruct with the original modifiers if possible
            replacement = variant_result.replacement
            
            # If original had modifiers, prepend them to replacement
            term_lower = term.lower()
            variant_lower = variant.lower()
            
            if term_lower != variant_lower:
                # Find what was stripped
                prefix = term_lower.replace(variant_lower, '').strip()
                if prefix:
                    # Prepend the modifier to the replacement
                    reconstructed = f"{prefix} {replacement}"
                    
                    # FILTER: Reject if reconstruction creates redundancy
                    # e.g., "CT scan of the chest" + "Scan cat" → "ct scan of the chest Scan cat" (redundant "scan")
                    # Check if prefix and replacement share significant words (overlap)
                    prefix_words = set(prefix.lower().split())
                    repl_words = set(replacement.lower().split())
                    overlap = prefix_words & repl_words
                    # Remove common stopwords from overlap check
                    stopwords = {'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with'}
                    significant_overlap = overlap - stopwords
                    
                    if significant_overlap:
                        # Redundant - reject this variant, try next
                        continue
                    
                    replacement = reconstructed
            
            return VerifiedReplacement(
                original=term,
                replacement=replacement,
                verified=True,
                source="UMLS",
                cui=variant_result.cui,
                semantic_type=variant_result.semantic_type
            )
    
    # All variants failed
    return result  # Return the original failure


# ============================================
# 2. TEMPORAL REPHRASING (Deterministic)
# ============================================

def get_temporal_replacement(term: str) -> VerifiedReplacement:
    """
    Convert temporal expressions using mathematical equivalence.
    
    Examples:
        "1-year history" → "12-month history"
        "2 weeks" → "14 days"
        "36 hours" → "1 day and 12 hours"
    """
    term_lower = term.lower().strip()
    
    # Conversion patterns (regex, converter function)
    patterns = [
        # Hyphenated: "X-year" → "X*12-month"
        (r'(\d+)-year', lambda m: f"{int(m.group(1)) * 12}-month"),
        (r'(\d+)-month', lambda m: f"{int(m.group(1)) * 4}-week" if int(m.group(1)) <= 3 else f"{int(m.group(1)) * 30}-day"),
        (r'(\d+)-week', lambda m: f"{int(m.group(1)) * 7}-day"),
        (r'(\d+)-day', lambda m: f"{int(m.group(1)) * 24}-hour"),
        
        # Hours: convert to days if >= 24
        (r'(\d+)-hour', lambda m: _hours_to_days(int(m.group(1)), hyphen=True)),
        
        # Spaced: "X years" → "X*12 months"
        (r'(\d+)\s+years?', lambda m: f"{int(m.group(1)) * 12} months"),
        (r'(\d+)\s+months?', lambda m: f"{int(m.group(1)) * 4} weeks" if int(m.group(1)) <= 3 else f"{int(m.group(1)) * 30} days"),
        (r'(\d+)\s+weeks?', lambda m: f"{int(m.group(1)) * 7} days"),
        (r'(\d+)\s+days?', lambda m: f"{int(m.group(1)) * 24} hours"),
        (r'(\d+)\s+hours?', lambda m: _hours_to_days(int(m.group(1)), hyphen=False)),
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, term_lower, re.IGNORECASE)
        if match:
            try:
                replacement_part = converter(match)
                new_term = re.sub(pattern, replacement_part, term_lower, count=1, flags=re.IGNORECASE)
                
                # Skip if no change
                if new_term.strip() == term_lower.strip():
                    continue
                
                return VerifiedReplacement(
                    original=term,
                    replacement=new_term,
                    verified=True,
                    source="temporal"
                )
            except Exception:
                continue
    
    return VerifiedReplacement(term, "", False, "temporal")


def _hours_to_days(hours: int, hyphen: bool = True) -> str:
    """Convert hours to days + hours format."""
    if hours >= 24:
        days = hours // 24
        remaining = hours % 24
        if remaining == 0:
            return f"{days}-day" if hyphen else f"{days} days"
        else:
            day_str = "day" if days == 1 else "days"
            hour_str = "hour" if remaining == 1 else "hours"
            return f"{days} {day_str} and {remaining} {hour_str}"
    else:
        # Keep as hours or convert to minutes
        minutes = hours * 60
        return f"{minutes}-minute" if hyphen else f"{minutes} minutes"


# ============================================
# 3. IRRELEVANT CORRELATION (Curated Lists)
# ============================================

# Curated non-clinical details that can be added/modified
IRRELEVANT_DETAILS = {
    "weather": [
        "It was a sunny day",
        "The weather was overcast",
        "It had been raining that morning",
    ],
    "day_of_week": [
        "on a Monday",
        "on a Friday afternoon",
        "early Tuesday morning",
    ],
    "waiting_room": [
        "after a brief wait",
        "The waiting room was busy",
        "There were several other patients waiting",
    ],
    "clothing": [
        "wearing casual clothes",
        "dressed in work attire",
        "wearing a hospital gown",
    ],
}

def get_irrelevant_correlation() -> VerifiedReplacement:
    """
    Get a random irrelevant detail to add to a note.
    These are non-clinical details that don't affect medical meaning.
    """
    import random
    
    category = random.choice(list(IRRELEVANT_DETAILS.keys()))
    detail = random.choice(IRRELEVANT_DETAILS[category])
    
    return VerifiedReplacement(
        original="",
        replacement=detail,
        verified=True,
        source="curated",
        semantic_type=category
    )


# ============================================
# 4. EQUIVALENT CITATION (RxNorm Drug Classes)
# ============================================

class RxNormLookup:
    """
    RxNorm API for drug → drug class lookups.
    No API key required.
    """
    
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"
    
    # Only accept these class types (pharmacologic classes)
    VALID_CLASS_TYPES = {"EPC"}  # Established Pharmacologic Class only
    
    # Reject these patterns in class names
    INVALID_PATTERNS = [
        'decreased', 'increased', 'reduced', 'elevated',  # Mechanisms
        'insufficiency', 'deficiency', 'disease', 'syndrome', 'disorder',  # Conditions
        'hypersensitivity', 'toxicity', 'adverse',  # Side effects
        ',',  # Formatting issues
    ]
    
    def __init__(self):
        self.session = requests.Session()
    
    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """Make cached API request."""
        key = _cache_key("rxnorm", url=url, params=params or {})
        if (cached := _cached(key)) is not None:
            return cached
        
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.ok:
                return _cache(key, resp.json())
        except Exception:
            pass
        return None
    
    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Get RxCUI for a drug name."""
        url = f"{self.BASE_URL}/rxcui.json"
        data = self._get(url, {"name": drug_name})
        if data:
            rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
            return rxcui_list[0] if rxcui_list else None
        return None
    
    def get_drug_class(self, drug_name: str) -> Optional[str]:
        """Get pharmacologic class for a drug."""
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return None
        
        url = f"{self.BASE_URL}/rxclass/class/byRxcui.json"
        data = self._get(url, {"rxcui": rxcui})
        if not data:
            return None
        
        class_list = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        
        for item in class_list:
            concept = item.get("rxclassMinConceptItem", {})
            class_type = concept.get("classType", "")
            class_name = concept.get("className", "")
            
            # Only accept EPC (Established Pharmacologic Class)
            if class_type not in self.VALID_CLASS_TYPES:
                continue
            
            # Reject invalid patterns
            class_lower = class_name.lower()
            if any(p in class_lower for p in self.INVALID_PATTERNS):
                continue
            
            # Must be different from drug name
            if class_lower == drug_name.lower():
                continue
            
            return class_name
        
        return None
    
    def get_drug_class_dict(self, drug_name: str) -> Optional[dict]:
        """
        Get drug class in old dict format for backward compatibility.
        
        Returns:
            {"drug": "metformin", "classes": ["Biguanide"], "rxcui": "..."}
        """
        rxcui = self.get_rxcui(drug_name)
        if not rxcui:
            return None
        
        drug_class = self.get_drug_class(drug_name)
        if drug_class:
            return {
                "drug": drug_name,
                "rxcui": rxcui,
                "classes": [drug_class]
            }
        return None


def get_equivalent_citation_replacement(term: str, term_type: str = "drug") -> VerifiedReplacement:
    """
    Get a verified equivalent citation replacement.
    
    For drugs: Returns pharmacologic drug class.
    
    Examples:
        "metformin" → "Biguanide" (drug class)
        "lisinopril" → "Angiotensin Converting Enzyme Inhibitor"
        "metoprolol" → "beta-Adrenergic Blocker"
    """
    term_clean = term.strip()
    term_lower = term_clean.lower()
    
    # Check cache
    key = _cache_key("equivalent_citation", term=term_lower, type=term_type)
    if (cached := _cached(key)) is not None:
        return cached
    
    if term_type == "drug":
        try:
            rxnorm = RxNormLookup()
            drug_class = rxnorm.get_drug_class(term_clean)
            
            if drug_class:
                # Match case of original term
                # If original is lowercase (e.g., "metformin"), make replacement lowercase
                if term_clean.islower():
                    drug_class = drug_class.lower()
                elif term_clean[0].islower():  # starts lowercase
                    drug_class = drug_class[0].lower() + drug_class[1:] if drug_class else drug_class
                
                return _cache(key, VerifiedReplacement(
                    original=term,
                    replacement=drug_class,
                    verified=True,
                    source="RxNorm",
                    semantic_type="Pharmacologic Class"
                ))
        except Exception:
            pass
    
    # Fallback: Try UMLS for procedures/non-drug medical terms
    # This handles cases like "Arthrocentesis", "Pap smear"
    # BUT: Reject imaging/scan procedures - they don't have good "equivalent citations"
    # e.g., "CT scan", "MRI", "ultrasound", "X-ray" should just fail
    imaging_keywords = ['scan', 'imaging', 'image', 'radiograph', 'tomography', 
                       'ultrasound', 'sonogram', 'angiogram', 'angiography',
                       ' mri', ' ct ', ' pet ', ' spect ', 'x-ray', 'xray',
                       'contrast', 'mammogram']
    term_lower_check = term_lower.lower()
    
    if not any(keyword in term_lower_check for keyword in imaging_keywords):
        umls_result = get_pseudo_factual_replacement_with_fallback(term)
        if umls_result.verified:
            return _cache(key, umls_result)
    
    return _cache(key, VerifiedReplacement(term, "", False, "RxNorm"))


# ============================================
# UNIFIED INTERFACE
# ============================================

def get_verified_replacement(term: str, change_type: str) -> VerifiedReplacement:
    """
    Unified interface for getting verified replacements.
    
    Args:
        term: The term to replace
        change_type: One of 'pseudo_factual', 'temporal_rephrasing', 
                     'irrelevant_correlation', 'equivalent_citation'
    
    Returns:
        VerifiedReplacement with verified=True if successful
    """
    if change_type == "pseudo_factual":
        # Use fallback version that tries term decomposition
        return get_pseudo_factual_replacement_with_fallback(term)
    
    elif change_type == "temporal_rephrasing":
        return get_temporal_replacement(term)
    
    elif change_type == "irrelevant_correlation":
        return get_irrelevant_correlation()
    
    elif change_type == "equivalent_citation":
        return get_equivalent_citation_replacement(term, term_type="drug")
    
    else:
        return VerifiedReplacement(term, "", False, "unknown")


# ============================================
# QUICK TEST
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Medical Knowledge Base v2 - Clean Design")
    print("=" * 60)
    
    # Test 1: Pseudo-Factual (UMLS)
    print("\n1. PSEUDO-FACTUAL SUBSTITUTION (UMLS CUI matching):")
    print("-" * 50)
    
    test_terms = ["hypertension", "diabetes", "pneumonia", "anxiety", "cirrhosis"]
    for term in test_terms:
        result = get_pseudo_factual_replacement(term)
        status = "✓" if result.verified else "✗"
        print(f"  {status} {term}")
        if result.verified:
            print(f"      → {result.replacement}")
            print(f"      CUI: {result.cui}, Type: {result.semantic_type}")
    
    # Test 2: Temporal
    print("\n2. TEMPORAL REPHRASING (Deterministic rules):")
    print("-" * 50)
    
    temporal_tests = ["1-year history", "2 weeks", "36-hour", "3 months", "7 days"]
    for term in temporal_tests:
        result = get_temporal_replacement(term)
        status = "✓" if result.verified else "✗"
        print(f"  {status} {term} → {result.replacement if result.verified else 'N/A'}")
    
    # Test 3: Equivalent Citation (RxNorm)
    print("\n3. EQUIVALENT CITATION (RxNorm drug classes):")
    print("-" * 50)
    
    drug_tests = ["metformin", "lisinopril", "metoprolol", "aspirin", "sertraline"]
    for term in drug_tests:
        result = get_equivalent_citation_replacement(term)
        status = "✓" if result.verified else "✗"
        print(f"  {status} {term}")
        if result.verified:
            print(f"      → {result.replacement}")
    
    # Test 4: Irrelevant Correlation
    print("\n4. IRRELEVANT CORRELATION (Curated lists):")
    print("-" * 50)
    
    for _ in range(3):
        result = get_irrelevant_correlation()
        print(f"  ✓ [{result.semantic_type}] {result.replacement}")
    
    print("\n" + "=" * 60)
    print("✓ All tests complete!")
    print("=" * 60)
