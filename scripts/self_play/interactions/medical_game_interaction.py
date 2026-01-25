"""Medical Error Detection Game Interaction for verl multi-turn RL training.

This interaction orchestrates the two-turn self-play game:
1. Turn 1 (Injector): Model receives a note and generates a modified version
2. Turn 2 (Assessor): Model classifies the modified note as CORRECT or INCORRECT

The Assessor sees ONLY the modified note (CoT is stripped from Injector output).
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

# Import verl's BaseInteraction
try:
    from verl.interactions.base import BaseInteraction
except ImportError:
    # Fallback for development/testing
    class BaseInteraction:
        def __init__(self, config: dict):
            self.config = config
            self.name = config.get("name", "interaction_agent")


class MedicalGameInteraction(BaseInteraction):
    """Two-turn medical error detection game interaction.
    
    Phase 1: Injector modifies a clinical note (benign or error injection)
    Phase 2: Assessor classifies the modified note (CORRECT or INCORRECT)
    
    Rewards are zero-sum:
    - Assessor correct: Assessor +1.0, Injector -1.0
    - Assessor wrong: Assessor -1.0, Injector +1.0
    - Format bonus: +0.2 for following output format
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._instance_dict = {}
        
        # Load prompts from config files
        self.detection_prompts = self._load_prompts(
            config.get("detection_prompts_path", 
                      "configs/prompts/error_detection_prompts.json")
        )
        
    def _load_prompts(self, path: str) -> dict:
        """Load prompts from JSON config file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    async def start_interaction(
        self, 
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        mode: Optional[str] = None,
        note_data: Optional[dict] = None,
        correct_note: Optional[str] = None,
        note_id: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """Initialize interaction session.
        
        Called by verl with interaction_kwargs from the dataset.
        See: https://verl.readthedocs.io/en/latest/sglang_multiturn/interaction_system.html
        
        Args:
            instance_id: Unique session ID (auto-generated if None)
            ground_truth: "CORRECT" or "INCORRECT" 
            mode: "benign" or "error_injection"
            note_data: Dict containing note information (legacy)
            correct_note: The original correct note text
            note_id: Unique identifier for this note
            error_type: Type of error (for error_injection mode)
            **kwargs: Additional kwargs passed from interaction_kwargs
            
        Returns:
            instance_id for this session
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Build note_data from kwargs if not provided directly
        if note_data is None:
            note_data = {
                "correct_note": correct_note or "",
                "note_id": note_id or "",
                "error_type": error_type or "",
            }
        
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth or kwargs.get("ground_truth"),
            "mode": mode or kwargs.get("mode"),
            "note_data": note_data,
            "injector_output": None,
            "generated_note": None,
            "assessor_output": None,
            "turn": 0,  # Track which turn we're on (incremented in generate_response)
        }
        
        return instance_id
    
    async def generate_response(
        self, 
        instance_id: str, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process model response and generate next prompt or terminate.
        
        Returns:
            (should_terminate, response_content, score, metadata)
        """
        instance = self._instance_dict[instance_id]
        instance["turn"] += 1  # Increment turn counter
        
        if instance["turn"] == 1:
            # Phase 1: Process Injector output
            return await self._process_injector_turn(instance_id, messages)
        elif instance["turn"] == 2:
            # Phase 2: Process Assessor output and compute final reward
            return await self._process_assessor_turn(instance_id, messages)
        else:
            # Should not reach here
            return True, "Error: Invalid turn number", 0.0, {}
    
    async def _process_injector_turn(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process Phase 1 (Injector) output and prepare Phase 2 (Assessor) prompt.
        
        Extracts the generated note from Injector output, strips hidden CoT,
        and constructs the Assessor's classification prompt.
        """
        instance = self._instance_dict[instance_id]
        
        # Extract Injector's response (last assistant message)
        injector_output = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                injector_output = msg.get("content", "")
                break
        
        instance["injector_output"] = injector_output
        
        # Parse the generated note (strip CoT)
        generated_note = self._extract_generated_note(injector_output)
        instance["generated_note"] = generated_note
        
        # Check if we got a valid note
        if not generated_note:
            # Penalize for invalid format
            return True, "Invalid output format. Expected 'generated_note:' section.", -1.0, {}
        
        # Construct Assessor prompt using the detection prompts
        assessor_prompt = self._construct_assessor_prompt(generated_note)
        
        # Move to turn 2
        instance["turn"] = 2
        
        # Return False (don't terminate), assessor_prompt, 0 score (intermediate)
        return False, assessor_prompt, 0.0, {"phase": "injector_complete"}
    
    async def _process_assessor_turn(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process Phase 2 (Assessor) output and compute final zero-sum reward."""
        instance = self._instance_dict[instance_id]
        
        # Extract Assessor's response
        assessor_output = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assessor_output = msg.get("content", "")
                break
        
        instance["assessor_output"] = assessor_output
        
        # Parse Assessor's classification
        assessor_answer = self._parse_assessor_answer(assessor_output)
        ground_truth = instance["ground_truth"]
        
        # Check format compliance
        has_valid_format = self._check_assessor_format(assessor_output)
        format_bonus = 0.2 if has_valid_format else 0.0
        
        # Compute zero-sum reward
        if assessor_answer == ground_truth:
            # Assessor wins
            assessor_reward = 1.0 + format_bonus
            feedback = f"Correct! The note was {ground_truth}."
        elif assessor_answer in ["CORRECT", "INCORRECT"]:
            # Assessor classified but was wrong (Injector wins)
            assessor_reward = -1.0 + format_bonus
            feedback = f"Incorrect. The note was {ground_truth}, but you classified it as {assessor_answer}."
        else:
            # Invalid format - penalize heavily
            assessor_reward = -1.0
            feedback = f"Invalid response format. Expected 'final_answer: \"CORRECT\"' or '\"INCORRECT\"'."
        
        # Game complete
        return True, feedback, assessor_reward, {
            "phase": "game_complete",
            "assessor_answer": assessor_answer,
            "ground_truth": ground_truth,
            "has_valid_format": has_valid_format,
        }
    
    def _extract_generated_note(self, injector_output: str) -> str:
        """Extract ONLY the generated_note from Injector output.
        
        CRITICAL FOR HIDDEN COT (SeRL paper design):
        Strips ALL of the following so Assessor cannot see:
        - <think>...</think> tags (CoT reasoning)
        - final_answer: "CORRECT" or "INCORRECT" (if present)
        - changes_made: {...} (metadata about what was changed)
        
        The Assessor should see ONLY the modified clinical note text,
        with no hints about whether it contains errors.
        
        Supports both:
        - v2 format: generated_note: ... final_answer: ... changes_made: ...
        - v3 format: <think>...</think> generated_note: ... (no final_answer)
        """
        # Step 1: Remove <think> tags and their content (Hidden CoT)
        output = re.sub(r'<think>.*?</think>', '', injector_output, flags=re.DOTALL)
        
        # Step 2: Extract ONLY the text after "generated_note:" 
        # Try multiple patterns to handle variations in formatting
        patterns = [
            # Pattern 1: v3 format - generated_note: until end (no final_answer)
            r'generated_note:\s*\n(.*?)$',
            # Pattern 2: v2 format - stop at final_answer or changes_made
            r'generated_note:\s*\n(.*?)(?=\n\s*final_answer:|\n\s*changes_made:|$)',
            # Pattern 3: More lenient - any text after generated_note:
            r'generated_note:\s*(.*?)(?=final_answer:|changes_made:|$)',
        ]
        
        extracted_note = None
        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_note = match.group(1).strip()
                if extracted_note and len(extracted_note) > 20:  # Non-empty, reasonable length
                    break
        
        if extracted_note:
            # Step 3: CRITICAL - Sanitize to remove any leaked information
            sanitized = self._sanitize_note_for_assessor(extracted_note)
            return sanitized
        
        # Fallback: If no generated_note: marker, use all text after stripping think tags
        # This handles edge cases where model doesn't follow format exactly
        clean_output = output.strip()
        if clean_output and len(clean_output) > 50:  # Has substantial content
            sanitized = self._sanitize_note_for_assessor(clean_output)
            if sanitized and len(sanitized) > 50:
                return sanitized
        
        # No valid extraction - return empty string (will trigger format penalty)
        return ""
    
    def _sanitize_note_for_assessor(self, note: str) -> str:
        """Remove any remaining leaked information from extracted note.
        
        This is a safety check to ensure no answer hints reach the Assessor.
        """
        if not note:
            return ""
        
        sanitized = note
        
        # Remove any remaining section markers that might have leaked
        sanitized = re.sub(r'final_answer:\s*["\']?(?:CORRECT|INCORRECT)["\']?', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'changes_made:\s*\{.*?\}', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r'changes_made:\s*$', '', sanitized, flags=re.IGNORECASE)
        
        # Remove any stray answer keywords at the end (common leak pattern)
        sanitized = re.sub(r'\n\s*"?(CORRECT|INCORRECT)"?\s*$', '', sanitized, flags=re.IGNORECASE)
        
        # Validate: Check for forbidden keywords that would leak the answer
        # If found after sanitization, something went wrong
        forbidden_patterns = [
            r'\bfinal_answer\b',
            r'\bchanges_made\b',
            r'\berror_type\b.*:',
            r'\bwords_changed\b',
            r'\boriginal_sentence\b',
            r'\bmodified_sentence\b',
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                # Log warning but continue - strip the problematic content
                sanitized = re.sub(pattern + r'.*', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized.strip()
    
    def _construct_assessor_prompt(self, generated_note: str) -> str:
        """Construct the Assessor's classification prompt.
        
        Uses the detection_prompts from error_detection_prompts.json.
        """
        user_template = self.detection_prompts.get("user_template", "")
        return user_template.format(note=generated_note)
    
    def _parse_assessor_answer(self, assessor_output: str) -> str:
        """Parse Assessor's final_answer from output.
        
        Expected format:
        final_answer: "CORRECT"
        or
        final_answer: "INCORRECT"
        """
        # Look for final_answer: "CORRECT" or "INCORRECT"
        match = re.search(
            r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?',
            assessor_output,
            re.IGNORECASE
        )
        
        if match:
            return match.group(1).upper()
        
        # Fallback: check for the words anywhere in output
        output_upper = assessor_output.upper()
        if "INCORRECT" in output_upper:
            return "INCORRECT"
        elif "CORRECT" in output_upper:
            return "CORRECT"
        
        return "UNKNOWN"
    
    def _check_assessor_format(self, assessor_output: str) -> bool:
        """Check if Assessor output follows the required format.
        
        Required:
        - final_answer: "CORRECT" or "INCORRECT"
        - Explanation: [some text]
        """
        has_final_answer = bool(re.search(
            r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?',
            assessor_output,
            re.IGNORECASE
        ))
        
        has_explanation = bool(re.search(
            r'Explanation:\s*\S+',
            assessor_output,
            re.IGNORECASE
        ))
        
        return has_final_answer and has_explanation
    
    async def calculate_score(
        self, 
        instance_id: str, 
        **kwargs
    ) -> float:
        """Calculate final score for the interaction.
        
        This is called by verl to get the reward for RL training.
        The actual reward computation happens in _process_assessor_turn.
        """
        instance = self._instance_dict.get(instance_id, {})
        
        # If we haven't completed the game, return 0
        if instance.get("turn", 1) != 2:
            return 0.0
        
        # The reward was already computed in _process_assessor_turn
        # We can reconstruct it here for consistency
        assessor_answer = self._parse_assessor_answer(
            instance.get("assessor_output", "")
        )
        ground_truth = instance.get("ground_truth", "")
        
        has_valid_format = self._check_assessor_format(
            instance.get("assessor_output", "")
        )
        format_bonus = 0.2 if has_valid_format else 0.0
        
        if assessor_answer == ground_truth:
            return 1.0 + format_bonus
        elif assessor_answer in ["CORRECT", "INCORRECT"]:
            return -1.0 + format_bonus
        else:
            return -1.0
    
    async def finalize_interaction(
        self, 
        instance_id: str, 
        **kwargs
    ) -> None:
        """Clean up resources for this interaction session."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
