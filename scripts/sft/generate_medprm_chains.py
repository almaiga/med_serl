"""
Generate Med-PRM structured reasoning chains using QwenMax API.
Processes MedEC pairs to create training data for Qwen3-4B/8B distillation.

Usage:
    python scripts/sft/generate_medprm_chains.py --limit 10  # Test with 10 pairs
    python scripts/sft/generate_medprm_chains.py             # Process all pairs
"""

import json
import os
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import argparse
import logging
import time

from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_async

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MedPRMChain:
    """Single Med-PRM reasoning chain."""
    note_id: str
    role: str  # 'assessor' or 'injector'
    scenario: str  # 'correct', 'incorrect', 'error', 'benign'
    input_note: str
    reasoning_chain: str
    label: str  # 'CORRECT' or 'INCORRECT'
    error_type: Optional[str] = None
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None
    change_type: Optional[str] = None  # benign change type: pseudo_factual, temporal_rephrasing, etc.
    is_valid: bool = True
    validation_errors: Optional[List[str]] = None
    token_estimate: int = 0


class ChainValidator:
    """Validate Med-PRM chain structure and content."""
    
    REQUIRED_SECTIONS = [
        "[CLINICAL FINDINGS]",
        "[CLINICAL PRINCIPLE]",
        "[REASONING]",
        "[ERROR_TYPE]",
        "[LOCATION]",
        "[CORRECTION]",
        "final_answer:"
    ]
    
    # Simple banned words - always forbidden in clinical reasoning sections
    BANNED_WORDS = [
        "possibly", "perhaps", "let me consider", 
        "i think", "it seems", "appears to", "seems to"
    ]
    
    # Hedging words - now only banned in FINAL CONCLUSION contexts
    # Allowed in differential diagnosis reasoning (e.g., "could suggest X, but...")
    HEDGING_WORDS_CONCLUSION = ["might", "could", "may"]
    
    # Patterns that indicate hedging on conclusions (BAD) vs differential reasoning (OK)
    # BAD: "the diagnosis might be", "this could be the cause"
    # OK: "could suggest malaria, but the findings rule it out"
    CONCLUSION_HEDGING_PATTERNS = [
        r"(the\s+)?diagnosis\s+(might|could|may)\s+be",
        r"(this|it)\s+(might|could|may)\s+be\s+(the\s+)?(cause|diagnosis|condition)",
        r"(might|could|may)\s+confirm",
        r"(might|could|may)\s+indicate\s+(that\s+)?the\s+diagnosis",
    ]
    
    # Hedging patterns (regex) - ban hedging uses of "suggests" but allow evidential
    HEDGING_PATTERNS = [
        r"suggests?\s+(that\s+)?(it\s+)?(might|could|may|possibly|perhaps)",
        r"(this|which)\s+(could|might|may)\s+suggest",
        r"suggests?\s+a\s+possible",
        r"could\s+suggest",
        r"may\s+suggest",
    ]
    
    # Error types that are eligible for retry (recoverable)
    RETRY_ELIGIBLE_ERRORS = [
        "Token count too high",
        "Banned phrase",
        "Hedging pattern",
        "Hedging word",
        "Hedging on conclusion"
    ]
    
    @classmethod
    def _extract_reasoning_sections(cls, reasoning_text: str) -> Dict[str, str]:
        """
        Parse [REASONING] into numbered steps.
        Returns dict: {'step1': '...', 'step2': '...', 'clinical_steps': '...'}
        """
        result = {'step1': '', 'clinical_steps': ''}
        
        # Try to find numbered steps (1. or 1:)
        step_pattern = r'(?:^|\n)\s*(?:1\.?|1:|WHY PLAUSIBLE:?)\s*(.+?)(?=(?:\n\s*(?:2\.?|2:)|$))'
        step1_match = re.search(step_pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
        
        if step1_match:
            result['step1'] = step1_match.group(1).strip()
            # Everything after step 1 is clinical reasoning
            step1_end = step1_match.end()
            result['clinical_steps'] = reasoning_text[step1_end:].strip()
        else:
            # No clear step structure - treat all as clinical
            result['clinical_steps'] = reasoning_text
        
        return result
    
    @classmethod
    def _check_hedging_section_aware(cls, chain: str) -> List[str]:
        """
        Section-aware hedging check:
        - Allow 'might/could/may' in differential reasoning (e.g., "could suggest X, but...")
        - Ban hedging only when it applies to CONCLUSIONS (diagnosis, final answer)
        - Always ban in [CLINICAL PRINCIPLE] section
        """
        errors = []
        
        # Check [CLINICAL PRINCIPLE] - no hedging allowed (this should be definitive)
        principle_match = re.search(r'\[CLINICAL PRINCIPLE\](.*?)\[REASONING\]', chain, re.DOTALL)
        if principle_match:
            principle_text = principle_match.group(1).lower()
            for word in cls.HEDGING_WORDS_CONCLUSION:
                if word in principle_text:
                    errors.append(f"Hedging word in CLINICAL PRINCIPLE: '{word}'")
                    return errors  # Return early on first error
        
        # Check [REASONING] section for conclusion-hedging patterns only
        reasoning_match = re.search(r'\[REASONING\](.*?)\[ERROR_TYPE\]', chain, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1)
            full_reasoning_lower = reasoning_text.lower()
            
            # Check for hedging on conclusions (BAD)
            for pattern in cls.CONCLUSION_HEDGING_PATTERNS:
                if re.search(pattern, full_reasoning_lower):
                    match = re.search(pattern, full_reasoning_lower)
                    errors.append(f"Hedging on conclusion: '{match.group()}'")
                    return errors
            
            # Check for banned words (always forbidden)
            for word in cls.BANNED_WORDS:
                if word in full_reasoning_lower:
                    errors.append(f"Banned phrase in REASONING: '{word}'")
                    return errors
            
            # Check for hedging patterns with "suggests" (still problematic)
            for pattern in cls.HEDGING_PATTERNS:
                if re.search(pattern, full_reasoning_lower):
                    match = re.search(pattern, full_reasoning_lower)
                    errors.append(f"Hedging pattern in REASONING: '{match.group()}'")
                    return errors
        
        # Check final_answer and Explanation lines
        final_answer_match = re.search(r'final_answer:.*?Explanation:.*', chain, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            conclusion_text = final_answer_match.group(0).lower()
            for word in cls.HEDGING_WORDS_CONCLUSION:
                if word in conclusion_text:
                    errors.append(f"Hedging word in final answer: '{word}'")
                    return errors
        
        return errors
    
    @classmethod
    def validate(cls, chain: str, expected_label: str, max_tokens: int = 768) -> Tuple[bool, List[str]]:
        """
        Validate chain structure and content.
        
        Args:
            chain: The generated reasoning chain
            expected_label: 'CORRECT' or 'INCORRECT'
            max_tokens: Maximum allowed tokens (soft limit)
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not chain:
            return False, ["Empty chain"]
        
        errors = []
        
        # Check required sections in order
        last_pos = -1
        for section in cls.REQUIRED_SECTIONS:
            pos = chain.find(section)
            if pos == -1:
                errors.append(f"Missing section: {section}")
            elif pos < last_pos:
                errors.append(f"Section out of order: {section}")
            else:
                last_pos = pos
        
        # Section-aware hedging check
        hedging_errors = cls._check_hedging_section_aware(chain)
        errors.extend(hedging_errors)
        
        # Check final_answer matches expected
        final_answer_match = re.search(r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?', chain, re.IGNORECASE)
        if final_answer_match:
            found_label = final_answer_match.group(1).upper()
            if found_label != expected_label:
                errors.append(f"final_answer mismatch: expected {expected_label}, found {found_label}")
        else:
            errors.append("Could not parse final_answer")
        
        # Token count with soft/hard limits
        estimated_tokens = cls.estimate_tokens(chain)
        hard_limit = int(max_tokens * 1.15)  # 15% buffer
        if estimated_tokens > hard_limit:
            errors.append(f"Token count too high: ~{estimated_tokens} (hard limit {hard_limit})")
        elif estimated_tokens > max_tokens:
            # Soft limit - warn but may still pass if no other errors
            errors.append(f"Token count too high: ~{estimated_tokens} (target {max_tokens})")
        
        # Check bullet points in CLINICAL FINDINGS (should have 1-4)
        findings_match = re.search(r'\[CLINICAL FINDINGS\](.*?)\[CLINICAL PRINCIPLE\]', chain, re.DOTALL)
        if findings_match:
            findings_text = findings_match.group(1)
            bullet_count = findings_text.count('•')
            if bullet_count == 0:
                errors.append("No bullet points in CLINICAL FINDINGS")
            elif bullet_count > 4:
                errors.append(f"Too many findings: {bullet_count} (max 4)")
        
        return len(errors) == 0, errors
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count (rough approximation: ~1.3 tokens per word)."""
        if not text:
            return 0
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    @classmethod
    def is_retry_eligible(cls, errors: List[str]) -> bool:
        """Check if validation errors are recoverable via retry."""
        if not errors:
            return False
        # Only retry if ALL errors are in the retry-eligible category
        for error in errors:
            is_eligible = any(eligible in error for eligible in cls.RETRY_ELIGIBLE_ERRORS)
            if not is_eligible:
                return False
        return True
    
    @classmethod
    def format_retry_feedback(cls, errors: List[str], max_tokens: int) -> str:
        """Format validation errors as feedback for retry prompt."""
        feedback_parts = ["\n\nPREVIOUS ATTEMPT FAILED. Fix these issues:"]
        
        for error in errors:
            if "Token count too high" in error:
                feedback_parts.append(
                    f"- OUTPUT TOO LONG (~{max_tokens} target). Be more concise: "
                    f"reduce [CLINICAL FINDINGS] to 3 bullets, keep [REASONING] focused."
                )
            elif "Hedging on conclusion" in error:
                feedback_parts.append(
                    f"- {error}. When stating the diagnosis, use definitive language: "
                    f"'indicates', 'confirms', 'demonstrates'. Avoid 'might/could/may' for conclusions."
                )
            elif "Hedging word in final answer" in error:
                feedback_parts.append(
                    f"- {error}. The final answer must be definitive, not hedged."
                )
            elif "Hedging word in CLINICAL PRINCIPLE" in error:
                feedback_parts.append(
                    f"- {error}. [CLINICAL PRINCIPLE] must be assertive medical fact, no hedging."
                )
            elif "Banned phrase" in error or "Hedging pattern" in error:
                feedback_parts.append(
                    f"- {error}. Use assertive language: 'indicates', 'confirms', 'demonstrates'."
                )
        
        feedback_parts.append("\nRegenerate with corrections:")
        return "\n".join(feedback_parts)


class QwenMaxClient:
    """QwenMax API client using OpenAI-compatible interface."""
    
    def __init__(
        self,
        model: str = "qwen-max",
        max_concurrent: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 3
    ):
        api_key = os.getenv('QWEN_API_KEY')
        if not api_key:
            raise ValueError("QWEN_API_KEY not found in .env file")
        
        # QwenMax uses DashScope OpenAI-compatible endpoint
        # For Singapore region (ap-southeast-1), use dashscope-intl
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Stats tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        logger.info(f"✓ Initialized QwenMax client")
        logger.info(f"  Model: {model}")
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  Temperature: {temperature}")
    
    async def generate(self, system_prompt: str, user_prompt: str, note_id: str) -> Optional[str]:
        """
        Generate a single chain with retry logic and rate limiting.
        
        Args:
            system_prompt: System message with chain structure instructions
            user_prompt: User message with clinical note
            note_id: Identifier for logging
            
        Returns:
            Generated chain text or None if failed
        """
        async with self.semaphore:
            self.total_calls += 1
            
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    result = response.choices[0].message.content
                    self.successful_calls += 1
                    return result
                    
                except RateLimitError as e:
                    wait_time = (2 ** attempt) * (1 + 0.1 * (attempt + 1))
                    logger.warning(f"Rate limit for {note_id}, retry {attempt+1} in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    
                except APIError as e:
                    wait_time = 2 ** attempt
                    logger.warning(f"API error for {note_id}: {e}, retry {attempt+1}")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Unexpected error for {note_id}: {e}")
                    if attempt == self.max_retries - 1:
                        self.failed_calls += 1
                        return None
                    await asyncio.sleep(2 ** attempt)
            
            self.failed_calls += 1
            return None
    
    def get_stats(self) -> Dict:
        """Get API call statistics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(self.total_calls, 1) * 100
        }


class MedPRMGenerator:
    """Main generator orchestrating Med-PRM chain creation."""
    
    def __init__(
        self,
        prompts_path: str,
        output_dir: str,
        model: str = "qwen-max",
        max_concurrent: int = 5
    ):
        self.client = QwenMaxClient(model=model, max_concurrent=max_concurrent)
        self.prompts = self._load_prompts(prompts_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Loaded prompts from {prompts_path}")
        logger.info(f"✓ Output directory: {output_dir}")
    
    def _load_prompts(self, path: str) -> Dict:
        """Load prompt templates from JSON file."""
        with open(path) as f:
            prompts = json.load(f)
        
        # Validate required keys
        required = ['assessor_correct', 'assessor_incorrect', 'injector_error', 'injector_benign']
        for key in required:
            if key not in prompts:
                raise ValueError(f"Missing prompt template: {key}")
            if 'system' not in prompts[key] or 'user_template' not in prompts[key]:
                raise ValueError(f"Prompt {key} missing 'system' or 'user_template'")
        
        # Load token limits from metadata (with defaults)
        self.token_limits = prompts.get('metadata', {}).get('token_limits', {
            'assessor_correct': 550,
            'assessor_incorrect': 600,
            'injector_error': 650,
            'injector_benign': 450
        })
        logger.info(f"  Token limits: {self.token_limits}")
        
        return prompts
    
    def _format_prompt(self, template: str, pair: Dict) -> str:
        """Format prompt template with pair data, handling missing keys gracefully."""
        return template.format(
            correct_note=pair.get('correct_note', ''),
            incorrect_note=pair.get('incorrect_note', ''),
            error_type=pair.get('error_type', ''),
            error_sentence=pair.get('error_sentence', ''),
            corrected_sentence=pair.get('corrected_sentence', '')
        )

    def _format_benign_prompt(self, template: str, record: Dict) -> str:
        """Format prompt template with benign change record data (different schema)."""
        return template.format(
            original_note=record.get('original_note', ''),
            modified_note=record.get('modified_note', ''),
            change_type=record.get('change_type', ''),
            original_term=record.get('original_term', ''),
            replacement_term=record.get('replacement_term', '')
        )
    
    async def generate_assessor_correct(self, pair: Dict) -> Optional[MedPRMChain]:
        """Generate Assessor-Correct chain with selective retry."""
        note_id = f"{pair['note_id']}_assessor_correct"
        max_tokens = self.token_limits.get('assessor_correct', 550)
        
        system = self.prompts['assessor_correct']['system']
        user = self._format_prompt(self.prompts['assessor_correct']['user_template'], pair)
        
        result = await self.client.generate(system, user, note_id)
        
        if not result:
            return None
        
        is_valid, errors = ChainValidator.validate(result, "CORRECT", max_tokens=max_tokens)
        
        # Selective retry: one attempt if errors are recoverable
        if not is_valid and ChainValidator.is_retry_eligible(errors):
            logger.info(f"Retrying {note_id} due to: {errors}")
            feedback = ChainValidator.format_retry_feedback(errors, max_tokens)
            retry_user = user + feedback
            
            result = await self.client.generate(system, retry_user, f"{note_id}_retry")
            if result:
                is_valid, errors = ChainValidator.validate(result, "CORRECT", max_tokens=max_tokens)
        
        return MedPRMChain(
            note_id=note_id,
            role="assessor",
            scenario="correct",
            input_note=pair['correct_note'],
            reasoning_chain=result,
            label="CORRECT",
            is_valid=is_valid,
            validation_errors=errors if errors else None,
            token_estimate=ChainValidator.estimate_tokens(result) if result else 0
        )
    
    async def generate_assessor_incorrect(self, pair: Dict) -> Optional[MedPRMChain]:
        """Generate Assessor-Incorrect chain with ground truth and selective retry."""
        note_id = f"{pair['note_id']}_assessor_incorrect"
        max_tokens = self.token_limits.get('assessor_incorrect', 600)
        
        system = self.prompts['assessor_incorrect']['system']
        user = self._format_prompt(self.prompts['assessor_incorrect']['user_template'], pair)
        
        result = await self.client.generate(system, user, note_id)
        
        if not result:
            return None
        
        is_valid, errors = ChainValidator.validate(result, "INCORRECT", max_tokens=max_tokens)
        
        # Selective retry: one attempt if errors are recoverable
        if not is_valid and ChainValidator.is_retry_eligible(errors):
            logger.info(f"Retrying {note_id} due to: {errors}")
            feedback = ChainValidator.format_retry_feedback(errors, max_tokens)
            retry_user = user + feedback
            
            result = await self.client.generate(system, retry_user, f"{note_id}_retry")
            if result:
                is_valid, errors = ChainValidator.validate(result, "INCORRECT", max_tokens=max_tokens)
        
        return MedPRMChain(
            note_id=note_id,
            role="assessor",
            scenario="incorrect",
            input_note=pair['incorrect_note'],
            reasoning_chain=result,
            label="INCORRECT",
            error_type=pair.get('error_type'),
            error_sentence=pair.get('error_sentence'),
            corrected_sentence=pair.get('corrected_sentence'),
            is_valid=is_valid,
            validation_errors=errors if errors else None,
            token_estimate=ChainValidator.estimate_tokens(result) if result else 0
        )
    
    async def generate_injector_error(self, pair: Dict) -> Optional[MedPRMChain]:
        """Generate Injector-Error chain (adversarial reasoning) with selective retry."""
        note_id = f"{pair['note_id']}_injector_error"
        max_tokens = self.token_limits.get('injector_error', 650)
        
        system = self.prompts['injector_error']['system']
        user = self._format_prompt(self.prompts['injector_error']['user_template'], pair)
        
        result = await self.client.generate(system, user, note_id)
        
        if not result:
            return None
        
        is_valid, errors = ChainValidator.validate(result, "INCORRECT", max_tokens=max_tokens)
        
        # Selective retry: one attempt if errors are recoverable
        if not is_valid and ChainValidator.is_retry_eligible(errors):
            logger.info(f"Retrying {note_id} due to: {errors}")
            feedback = ChainValidator.format_retry_feedback(errors, max_tokens)
            retry_user = user + feedback
            
            result = await self.client.generate(system, retry_user, f"{note_id}_retry")
            if result:
                is_valid, errors = ChainValidator.validate(result, "INCORRECT", max_tokens=max_tokens)
        
        return MedPRMChain(
            note_id=note_id,
            role="injector",
            scenario="error",
            input_note=pair['correct_note'],
            reasoning_chain=result,
            label="INCORRECT",
            error_type=pair.get('error_type'),
            error_sentence=pair.get('error_sentence'),
            corrected_sentence=pair.get('corrected_sentence'),
            is_valid=is_valid,
            validation_errors=errors if errors else None,
            token_estimate=ChainValidator.estimate_tokens(result) if result else 0
        )

    async def generate_injector_benign(self, record: Dict) -> Optional[MedPRMChain]:
        """Generate Injector-Benign chain (semantic equivalence reasoning).
        
        Uses benign change schema: original_note, modified_note, change_type,
        original_term, replacement_term.
        """
        note_id = f"{record.get('note_id', 'unknown')}_{record.get('change_type', 'benign')}_injector_benign"
        max_tokens = self.token_limits.get('injector_benign', 450)

        system = self.prompts['injector_benign']['system']
        user = self._format_benign_prompt(self.prompts['injector_benign']['user_template'], record)

        result = await self.client.generate(system, user, note_id)

        if not result:
            return None

        is_valid, errors = ChainValidator.validate(result, "CORRECT", max_tokens=max_tokens)

        # Selective retry: one attempt if errors are recoverable
        if not is_valid and ChainValidator.is_retry_eligible(errors):
            logger.info(f"Retrying {note_id} due to: {errors}")
            feedback = ChainValidator.format_retry_feedback(errors, max_tokens)
            result = await self.client.generate(system, user + feedback, f"{note_id}_retry")
            if result:
                is_valid, errors = ChainValidator.validate(result, "CORRECT", max_tokens=max_tokens)

        return MedPRMChain(
            note_id=note_id,
            role="injector",
            scenario="benign",
            input_note=record.get('original_note', ''),
            reasoning_chain=result,
            label="CORRECT",
            error_type=None,
            error_sentence=None,
            corrected_sentence=None,
            change_type=record.get('change_type', 'pseudo_factual'),
            is_valid=is_valid,
            validation_errors=errors if errors else None,
            token_estimate=ChainValidator.estimate_tokens(result) if result else 0
        )
    
    async def generate_chains_for_pair(self, pair: Dict, scenarios: List[str] = None) -> List[MedPRMChain]:
        """Generate chain types for a single medec pair based on selected scenarios.
        
        Note: injector_benign is NOT handled here - it uses a different data schema
        and is processed via process_benign_dataset() instead.
        
        Args:
            pair: Note pair data (medec schema: correct_note, incorrect_note, etc.)
            scenarios: List of scenarios to generate. Options:
                       'assessor_correct', 'assessor_incorrect', 'injector_error'
                       If None, generates all medec scenarios.
        """
        if scenarios is None:
            scenarios = ['assessor_correct', 'assessor_incorrect', 'injector_error']
        
        chains = []
        tasks = []
        
        if 'assessor_correct' in scenarios:
            tasks.append(self.generate_assessor_correct(pair))
        if 'assessor_incorrect' in scenarios:
            tasks.append(self.generate_assessor_incorrect(pair))
        if 'injector_error' in scenarios:
            tasks.append(self.generate_injector_error(pair))
        
        if not tasks:
            return chains
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, MedPRMChain):
                chains.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Exception for {pair['note_id']}: {result}")
        
        return chains
    
    async def process_dataset(
        self,
        pairs: List[Dict],
        checkpoint_every: int = 25,
        resume_from: int = 0,
        scenarios: List[str] = None
    ) -> List[MedPRMChain]:
        """
        Process all pairs with checkpointing and progress tracking.
        
        Args:
            pairs: List of note pairs to process
            checkpoint_every: Save checkpoint every N pairs
            resume_from: Resume from this pair index
            scenarios: List of scenarios to generate (None = all)
            
        Returns:
            List of all generated chains
        """
        all_chains = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"medprm_chains_{timestamp}.jsonl"
        checkpoint_file = self.output_dir / "checkpoint.json"
        stats_file = self.output_dir / f"stats_{timestamp}.json"
        
        # Handle resume
        if resume_from > 0:
            logger.info(f"Resuming from pair index {resume_from}")
            pairs = pairs[resume_from:]
        
        logger.info(f"Processing {len(pairs)} pairs")
        logger.info(f"Output: {output_file}")
        logger.info(f"Scenarios: {scenarios or 'all'}")
        chains_per_pair = len(scenarios) if scenarios else 3
        logger.info(f"Expected chains: ~{len(pairs) * chains_per_pair}")
        
        start_time = time.time()
        valid_count = 0
        error_type_counts = {}
        
        # Process pairs with progress bar
        for i, pair in enumerate(tqdm_async(pairs, desc="Generating chains")):
            chains = await self.generate_chains_for_pair(pair, scenarios=scenarios)
            all_chains.extend(chains)
            
            # Track stats
            for chain in chains:
                if chain.is_valid:
                    valid_count += 1
                if chain.error_type:
                    error_type_counts[chain.error_type] = error_type_counts.get(chain.error_type, 0) + 1
            
            # Write incrementally to JSONL
            with open(output_file, 'a') as f:
                for chain in chains:
                    f.write(json.dumps(asdict(chain)) + '\n')
            
            # Checkpoint every N pairs
            if (i + 1) % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60  # pairs per minute
                
                checkpoint_data = {
                    'last_index': resume_from + i,
                    'total_chains': len(all_chains),
                    'valid_chains': valid_count,
                    'elapsed_seconds': elapsed,
                    'pairs_per_minute': rate,
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                logger.info(f"Checkpoint: {i+1}/{len(pairs)} pairs | {len(all_chains)} chains | {rate:.1f} pairs/min")
        
        # Final statistics
        elapsed = time.time() - start_time
        api_stats = self.client.get_stats()
        
        final_stats = {
            'total_pairs': len(pairs),
            'total_chains': len(all_chains),
            'valid_chains': valid_count,
            'invalid_chains': len(all_chains) - valid_count,
            'validation_rate': valid_count / max(len(all_chains), 1) * 100,
            'error_type_distribution': error_type_counts,
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'chains_per_minute': len(all_chains) / elapsed * 60,
            'api_stats': api_stats,
            'output_file': str(output_file),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total pairs processed: {len(pairs)}")
        logger.info(f"Total chains generated: {len(all_chains)}")
        logger.info(f"Valid chains: {valid_count} ({final_stats['validation_rate']:.1f}%)")
        logger.info(f"Invalid chains: {len(all_chains) - valid_count}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"API success rate: {api_stats['success_rate']:.1f}%")
        logger.info(f"Output: {output_file}")
        logger.info(f"Stats: {stats_file}")
        
        return all_chains

    async def process_benign_dataset(
        self,
        records: List[Dict],
        checkpoint_every: int = 25,
        resume_from: int = 0,
    ) -> List[MedPRMChain]:
        """
        Process benign change records to generate injector_benign CoT chains.
        Reads from benign_train_clean.jsonl schema (original_note/modified_note).
        """
        all_chains = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"medprm_chains_benign_{timestamp}.jsonl"
        checkpoint_file = self.output_dir / "checkpoint_benign.json"
        stats_file = self.output_dir / f"stats_benign_{timestamp}.json"

        if resume_from > 0:
            logger.info(f"Resuming benign from record index {resume_from}")
            records = records[resume_from:]

        logger.info(f"Processing {len(records)} benign change records")
        logger.info(f"Output: {output_file}")

        start_time = time.time()
        valid_count = 0
        change_type_counts = {}

        for i, record in enumerate(tqdm_async(records, desc="Generating injector_benign chains")):
            chain = await self.generate_injector_benign(record)

            if chain is None:
                continue

            all_chains.append(chain)

            if chain.is_valid:
                valid_count += 1

            ct = record.get('change_type', 'unknown')
            change_type_counts[ct] = change_type_counts.get(ct, 0) + 1

            # Write incrementally
            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(chain)) + '\n')

            # Checkpoint every N records
            if (i + 1) % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                checkpoint_data = {
                    'last_index': resume_from + i,
                    'total_chains': len(all_chains),
                    'valid_chains': valid_count,
                    'elapsed_seconds': elapsed,
                    'records_per_minute': rate,
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"Checkpoint: {i+1}/{len(records)} records | {valid_count} valid | {rate:.1f}/min")

        # Final statistics
        elapsed = time.time() - start_time
        api_stats = self.client.get_stats()

        final_stats = {
            'total_records': len(records),
            'total_chains': len(all_chains),
            'valid_chains': valid_count,
            'invalid_chains': len(all_chains) - valid_count,
            'validation_rate': valid_count / max(len(all_chains), 1) * 100,
            'change_type_distribution': change_type_counts,
            'elapsed_seconds': elapsed,
            'elapsed_minutes': elapsed / 60,
            'chains_per_minute': len(all_chains) / max(elapsed, 1) * 60,
            'api_stats': api_stats,
            'output_file': str(output_file),
            'timestamp': datetime.now().isoformat()
        }

        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)

        logger.info("=" * 60)
        logger.info("BENIGN GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Records processed: {len(records)}")
        logger.info(f"Chains generated: {len(all_chains)}")
        logger.info(f"Valid chains: {valid_count} ({final_stats['validation_rate']:.1f}%)")
        logger.info(f"Change type distribution: {change_type_counts}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"API success rate: {api_stats['success_rate']:.1f}%")
        logger.info(f"Output: {output_file}")

        return all_chains


def load_pairs(input_path: str) -> List[Dict]:
    """Load pairs from JSONL file."""
    pairs = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Med-PRM reasoning chains using QwenMax API"
    )
    parser.add_argument(
        '--input',
        default='data_processed/medec_paired/train_val_split/sft_train.jsonl',
        help='Input JSONL file with note pairs'
    )
    parser.add_argument(
        '--prompts',
        default='configs/prompts/sft/medprm_prompts.json',
        help='Prompt templates JSON file'
    )
    parser.add_argument(
        '--output',
        default='data_processed/medprm_chains',
        help='Output directory for generated chains'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of pairs to process (for testing)'
    )
    parser.add_argument(
        '--resume',
        type=int,
        default=0,
        help='Resume from this pair index'
    )
    parser.add_argument(
        '--model',
        default='qwen-max',
        help='QwenMax model to use'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='Max concurrent API calls'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=25,
        help='Save checkpoint every N pairs'
    )
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['assessor_correct', 'assessor_incorrect', 'injector_error', 'injector_benign', 'all'],
        default=['all'],
        help='Scenarios to generate. "all" runs all medec + benign scenarios. '
             'injector_benign uses --benign-input instead of --input.'
    )
    parser.add_argument(
        '--benign-input',
        default='data_processed/benign_changes/benign_train_clean.jsonl',
        help='Input JSONL for injector_benign scenario (benign changes schema)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MedPRMGenerator(
        prompts_path=args.prompts,
        output_dir=args.output,
        model=args.model,
        max_concurrent=args.concurrency
    )
    
    # Determine which pipelines to run
    run_medec = False
    run_benign = False
    medec_scenarios = None  # None means all medec scenarios
    
    if args.scenarios and 'all' not in args.scenarios:
        # Specific scenarios selected
        if 'injector_benign' in args.scenarios:
            run_benign = True
        medec_only = [s for s in args.scenarios if s != 'injector_benign']
        if medec_only:
            run_medec = True
            medec_scenarios = medec_only
    else:
        # 'all' selected — run everything
        run_medec = True
        run_benign = True
    
    # Process medec pairs (assessor_correct, assessor_incorrect, injector_error)
    if run_medec:
        logger.info(f"Loading medec pairs from {args.input}")
        pairs = load_pairs(args.input)
        logger.info(f"Loaded {len(pairs)} medec pairs")
        
        if args.limit:
            pairs = pairs[:args.limit]
            logger.info(f"Limited to {len(pairs)} pairs")
        
        logger.info(f"Medec scenarios: {medec_scenarios or 'all'}")
        await generator.process_dataset(
            pairs,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume,
            scenarios=medec_scenarios
        )
    
    # Process benign changes (injector_benign)
    if run_benign:
        benign_path = Path(args.benign_input)
        if not benign_path.exists():
            logger.error(f"Benign input not found: {benign_path}")
            logger.error("Run the verify+fix pipeline first, then create benign_train_clean.jsonl")
        else:
            logger.info(f"Loading benign records from {benign_path}")
            benign_records = load_pairs(str(benign_path))
            logger.info(f"Loaded {len(benign_records)} benign records")
            
            if args.limit:
                benign_records = benign_records[:args.limit]
                logger.info(f"Limited to {len(benign_records)} benign records")
            
            await generator.process_benign_dataset(
                benign_records,
                checkpoint_every=args.checkpoint_every,
                resume_from=args.resume,
            )


if __name__ == '__main__':
    asyncio.run(main())
