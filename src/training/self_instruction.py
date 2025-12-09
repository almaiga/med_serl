"""
Self-Instruction Module for MedSeRL.

Implements the Scribe Agent's ability to generate NEW clinical notes
with injected errors, expanding the training distribution beyond MEDEC.

The Scribe generates:
1. A clinical scenario (patient presentation, findings, diagnosis)
2. An intentional error injection (wrong diagnosis, medication, etc.)
3. Ground truth metadata (what error was injected, where)

This enables training on diverse synthetic cases while maintaining
deterministic rewards (we know what error was injected).
"""

import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# Error types that can be injected
ERROR_TYPES = [
    "diagnosis",
    "pharmacotherapy",
    "management",
    "treatment",
    "causalOrganism"
]

# Clinical scenarios for generation
CLINICAL_SCENARIOS = [
    {
        "category": "cardiology",
        "presentations": [
            "chest pain", "shortness of breath", "palpitations",
            "syncope", "leg swelling", "fatigue"
        ],
        "conditions": [
            "myocardial infarction", "heart failure", "atrial fibrillation",
            "pulmonary embolism", "aortic dissection", "pericarditis"
        ]
    },
    {
        "category": "infectious_disease",
        "presentations": [
            "fever", "cough", "dysuria", "rash", "joint pain", "confusion"
        ],
        "conditions": [
            "pneumonia", "urinary tract infection", "cellulitis",
            "meningitis", "sepsis", "endocarditis"
        ]
    },
    {
        "category": "neurology",
        "presentations": [
            "headache", "weakness", "numbness", "vision changes",
            "seizure", "altered mental status"
        ],
        "conditions": [
            "stroke", "migraine", "multiple sclerosis", "Parkinson's disease",
            "epilepsy", "Guillain-Barré syndrome"
        ]
    },
    {
        "category": "gastroenterology",
        "presentations": [
            "abdominal pain", "nausea", "vomiting", "diarrhea",
            "bloody stool", "jaundice"
        ],
        "conditions": [
            "appendicitis", "cholecystitis", "pancreatitis",
            "inflammatory bowel disease", "GI bleeding", "hepatitis"
        ]
    },
    {
        "category": "pulmonology",
        "presentations": [
            "cough", "dyspnea", "wheezing", "hemoptysis", "chest pain"
        ],
        "conditions": [
            "asthma", "COPD", "pneumonia", "pulmonary embolism",
            "lung cancer", "tuberculosis"
        ]
    }
]

# Common error substitutions by type
ERROR_SUBSTITUTIONS = {
    "diagnosis": [
        ("myocardial infarction", "gastroesophageal reflux"),
        ("pulmonary embolism", "anxiety attack"),
        ("stroke", "migraine"),
        ("appendicitis", "gastroenteritis"),
        ("meningitis", "viral syndrome"),
        ("aortic dissection", "musculoskeletal pain"),
        ("sepsis", "viral infection"),
        ("heart failure", "deconditioning"),
    ],
    "pharmacotherapy": [
        ("metformin", "glipizide"),  # Different diabetes meds
        ("lisinopril", "amlodipine"),  # ACE-I vs CCB
        ("warfarin", "aspirin"),  # Anticoagulant vs antiplatelet
        ("ceftriaxone", "azithromycin"),  # Different antibiotics
        ("morphine", "ibuprofen"),  # Opioid vs NSAID
        ("insulin", "metformin"),  # Injectable vs oral
        ("heparin", "enoxaparin"),  # UFH vs LMWH
        ("vancomycin", "cefazolin"),  # MRSA coverage vs not
    ],
    "management": [
        ("emergent surgery", "observation"),
        ("ICU admission", "discharge home"),
        ("immediate intervention", "watchful waiting"),
        ("cardiac catheterization", "stress test"),
        ("CT scan", "ultrasound"),
        ("blood transfusion", "iron supplementation"),
        ("intubation", "supplemental oxygen"),
    ],
    "treatment": [
        ("thrombolysis", "antiplatelet therapy"),
        ("surgical intervention", "conservative management"),
        ("IV antibiotics", "oral antibiotics"),
        ("dialysis", "fluid restriction"),
        ("cardioversion", "rate control"),
        ("mechanical ventilation", "BiPAP"),
    ],
    "causalOrganism": [
        ("Staphylococcus aureus", "Streptococcus pneumoniae"),
        ("E. coli", "Klebsiella"),
        ("Pseudomonas", "Enterococcus"),
        ("Mycobacterium tuberculosis", "Mycoplasma pneumoniae"),
        ("Neisseria meningitidis", "viral etiology"),
        ("Clostridium difficile", "viral gastroenteritis"),
    ]
}


@dataclass
class SyntheticCase:
    """A synthetically generated clinical case with injected error."""
    clinical_note: str
    has_error: bool
    error_type: Optional[str]
    error_description: Optional[str]
    correct_value: Optional[str]
    incorrect_value: Optional[str]
    source: str = "self_instruction"


def get_self_instruction_prompt(
    error_type: str,
    scenario: Optional[Dict] = None
) -> str:
    """
    Generate a prompt for the Scribe to create a clinical note with an error.
    
    Args:
        error_type: Type of error to inject
        scenario: Optional clinical scenario to use
        
    Returns:
        Prompt string for the Scribe agent
    """
    if scenario is None:
        scenario = random.choice(CLINICAL_SCENARIOS)
    
    category = scenario["category"]
    presentation = random.choice(scenario["presentations"])
    condition = random.choice(scenario["conditions"])
    
    # Get a relevant substitution for this error type
    substitutions = ERROR_SUBSTITUTIONS.get(error_type, [])
    if substitutions:
        correct, incorrect = random.choice(substitutions)
    else:
        correct, incorrect = "correct_term", "incorrect_term"
    
    prompt = f"""Generate a realistic clinical note for a patient case.

Clinical Context:
- Category: {category}
- Chief complaint: {presentation}
- Likely condition: {condition}

Task: Write a clinical note that contains ONE {error_type} error.
The note should use "{incorrect}" where "{correct}" would be appropriate.

Requirements:
1. Write 3-5 sentences describing the patient presentation
2. Include relevant vital signs and exam findings
3. State the diagnosis/assessment
4. Include the treatment plan with the intentional error
5. Make the error subtle but clinically significant

Write only the clinical note, no explanations:"""

    return prompt, correct, incorrect


def get_clean_note_prompt(scenario: Optional[Dict] = None) -> str:
    """
    Generate a prompt for the Scribe to create a CORRECT clinical note.
    
    Args:
        scenario: Optional clinical scenario to use
        
    Returns:
        Prompt string for the Scribe agent
    """
    if scenario is None:
        scenario = random.choice(CLINICAL_SCENARIOS)
    
    category = scenario["category"]
    presentation = random.choice(scenario["presentations"])
    condition = random.choice(scenario["conditions"])
    
    prompt = f"""Generate a realistic, medically accurate clinical note.

Clinical Context:
- Category: {category}
- Chief complaint: {presentation}
- Condition: {condition}

Requirements:
1. Write 3-5 sentences describing the patient presentation
2. Include relevant vital signs and exam findings
3. State the correct diagnosis/assessment
4. Include an appropriate treatment plan
5. Ensure all medical information is accurate

Write only the clinical note, no explanations:"""

    return prompt


def generate_self_instruction_batch(
    batch_size: int = 8,
    error_ratio: float = 0.5
) -> List[Dict]:
    """
    Generate a batch of self-instruction prompts for the Scribe.
    
    Args:
        batch_size: Number of prompts to generate
        error_ratio: Fraction of prompts that should have errors
        
    Returns:
        List of prompt dictionaries with metadata
    """
    prompts = []
    num_errors = int(batch_size * error_ratio)
    
    # Generate error cases
    for i in range(num_errors):
        error_type = random.choice(ERROR_TYPES)
        scenario = random.choice(CLINICAL_SCENARIOS)
        prompt, correct, incorrect = get_self_instruction_prompt(
            error_type, scenario
        )
        
        prompts.append({
            "scribe_prompt": prompt,
            "has_error": True,
            "error_type": error_type,
            "correct_value": correct,
            "incorrect_value": incorrect,
            "scenario": scenario["category"],
            "source": "self_instruction_error"
        })
    
    # Generate clean cases
    for i in range(batch_size - num_errors):
        scenario = random.choice(CLINICAL_SCENARIOS)
        prompt = get_clean_note_prompt(scenario)
        
        prompts.append({
            "scribe_prompt": prompt,
            "has_error": False,
            "error_type": None,
            "correct_value": None,
            "incorrect_value": None,
            "scenario": scenario["category"],
            "source": "self_instruction_clean"
        })
    
    random.shuffle(prompts)
    return prompts


def process_scribe_output(
    generated_text: str,
    prompt_metadata: Dict
) -> SyntheticCase:
    """
    Process Scribe output into a SyntheticCase with ground truth.
    
    Args:
        generated_text: The clinical note generated by Scribe
        prompt_metadata: Metadata from the generation prompt
        
    Returns:
        SyntheticCase with ground truth labels
    """
    return SyntheticCase(
        clinical_note=generated_text.strip(),
        has_error=prompt_metadata["has_error"],
        error_type=prompt_metadata.get("error_type"),
        error_description=f"Used {prompt_metadata.get('incorrect_value')} "
                         f"instead of {prompt_metadata.get('correct_value')}"
                         if prompt_metadata["has_error"] else None,
        correct_value=prompt_metadata.get("correct_value"),
        incorrect_value=prompt_metadata.get("incorrect_value"),
        source=prompt_metadata.get("source", "self_instruction")
    )


def build_ground_truth(case: SyntheticCase) -> Dict:
    """
    Build ground truth dictionary from a SyntheticCase.
    
    Args:
        case: The synthetic case
        
    Returns:
        Ground truth dict compatible with reward_engine
    """
    return {
        "has_error": case.has_error,
        "error_type": case.error_type,
        "source": case.source
    }


def get_verification_prompt(
    clinical_note: str,
    expected_error: bool,
    error_type: Optional[str] = None,
    incorrect_value: Optional[str] = None,
    correct_value: Optional[str] = None
) -> str:
    """
    Generate a verification prompt to check if Scribe followed instructions.
    
    Args:
        clinical_note: The generated clinical note
        expected_error: Whether we asked for an error
        error_type: Type of error we asked for
        incorrect_value: The wrong term we asked Scribe to use
        correct_value: The correct term that should have been used
        
    Returns:
        Verification prompt string
    """
    if expected_error:
        prompt = f"""Verify if this clinical note contains the specified error.

Clinical Note:
{clinical_note}

Expected Error:
- Type: {error_type}
- The note should incorrectly use "{incorrect_value}" where "{correct_value}" would be medically appropriate.

Question: Does this note contain the specified error (using "{incorrect_value}" incorrectly)?

Answer with only YES or NO:"""
    else:
        prompt = f"""Verify if this clinical note is medically accurate.

Clinical Note:
{clinical_note}

Question: Is this clinical note free of obvious medical errors in diagnosis, treatment, or medication?

Answer with only YES or NO:"""
    
    return prompt


def parse_verification_response(response: str) -> bool:
    """
    Parse the verification model's response.
    
    Args:
        response: Model response to verification prompt
        
    Returns:
        True if verification passed, False otherwise
    """
    response_lower = response.strip().lower()
    
    # Look for YES/NO
    if response_lower.startswith("yes"):
        return True
    if response_lower.startswith("no"):
        return False
    
    # Fallback: check for presence of yes/no anywhere
    if "yes" in response_lower and "no" not in response_lower:
        return True
    if "no" in response_lower and "yes" not in response_lower:
        return False
    
    # Ambiguous - default to failed verification
    return False


def verify_self_instruction(
    model,
    tokenizer,
    clinical_note: str,
    prompt_metadata: Dict,
    device: str = "cuda"
) -> Tuple[bool, str]:
    """
    Verify that a self-instructed sample has the expected properties.
    
    Uses the model itself to verify whether:
    - Error samples actually contain the requested error
    - Clean samples are actually error-free
    
    Args:
        model: The language model for verification
        tokenizer: Tokenizer for the model
        clinical_note: The generated clinical note
        prompt_metadata: Metadata from generation (has_error, error_type, etc.)
        device: Device to run on
        
    Returns:
        Tuple of (verification_passed, verification_response)
    """
    import torch
    
    # Build verification prompt
    verification_prompt = get_verification_prompt(
        clinical_note=clinical_note,
        expected_error=prompt_metadata["has_error"],
        error_type=prompt_metadata.get("error_type"),
        incorrect_value=prompt_metadata.get("incorrect_value"),
        correct_value=prompt_metadata.get("correct_value")
    )
    
    # Tokenize
    inputs = tokenizer(
        verification_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    
    # Generate verification response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Just need YES/NO
            do_sample=False,  # Deterministic for verification
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    # Parse response
    verified = parse_verification_response(response)
    
    # For error samples: YES means error is present (good)
    # For clean samples: YES means it's clean (good)
    return verified, response.strip()


@dataclass
class VerifiedSample:
    """A verified self-instructed sample ready for training."""
    clinical_note: str
    ground_truth: Dict
    quadrant: str
    verified: bool
    verification_response: str
