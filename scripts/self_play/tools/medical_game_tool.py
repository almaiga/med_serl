"""Medical Error Detection Game Tool for verl multi-turn rollout.

This tool orchestrates the two-turn self-play game:
1. Turn 1 (Injector): Model receives a note and generates output
2. Turn 2 (Assessor): Model classifies the generated note

Prompts are loaded from config files, not hardcoded.
"""

import json
import random
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Assuming verl's tool base class interface
try:
    from verl.tools.base_tool import BaseTool
except ImportError:
    class BaseTool:
        def __init__(self, config: dict): 
            self.config = config
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

import sys
sys.path.append("scripts/self_play")
from cot_parser import extract_public_response, parse_injector_output


class GameMode(Enum):
    BENIGN = "benign"
    ERROR_INJECTION = "error_injection"


@dataclass
class GameState:
    """Tracks state across turns in the self-play game."""
    mode: GameMode
    turn: int = 0
    note_data: dict = field(default_factory=dict)
    injector_output: Optional[str] = None
    injector_parsed: Optional[Any] = None
    generated_note: Optional[str] = None
    ground_truth: Optional[str] = None  # "CORRECT" or "INCORRECT"
    game_complete: bool = False


class PromptLoader:
    """Loads prompts from JSON config files."""
    
    def __init__(
        self,
        injection_prompts_path: str = "configs/prompts/error_injection_prompts_v2.json",
        detection_prompts_path: str = "configs/prompts/error_detection_prompts.json",
    ):
        self.injection_prompts = self._load_json(injection_prompts_path)
        self.detection_prompts = self._load_json(detection_prompts_path)
    
    def _load_json(self, path: str) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_injector_system_prompt(self, mode: GameMode) -> str:
        """Get system prompt for Injector based on mode."""
        if mode == GameMode.BENIGN:
            return self.injection_prompts["system_prompt_correct"]
        else:
            return self.injection_prompts["system_prompt_incorrect"]
    
    def get_injector_user_prompt(self, mode: GameMode, note_data: dict) -> str:
        """Get user prompt for Injector based on mode."""
        if mode == GameMode.BENIGN:
            template = self.injection_prompts["injector_correct_template"]
            return template.format(note=note_data["correct_note"])
        else:
            template = self.injection_prompts["injector_incorrect_template"]
            return template.format(
                note=note_data["correct_note"],
                prompt_intent=note_data.get("error_type", "clinical error"),
            )
    
    def get_assessor_system_prompt(self) -> str:
        """Get system prompt for Assessor."""
        return self.detection_prompts["system_prompt"]
    
    def get_assessor_user_prompt(self, note: str) -> str:
        """Get user prompt for Assessor."""
        template = self.detection_prompts["user_template"]
        return template.format(note=note)


class MedicalGameTool(BaseTool):
    """Two-turn medical error detection game tool."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.benign_ratio = config.get("benign_ratio", 0.5)
        
        # Load prompts from config files
        self.prompt_loader = PromptLoader(
            injection_prompts_path=config.get(
                "injection_prompts_path",
                "configs/prompts/error_injection_prompts_v2.json"
            ),
            detection_prompts_path=config.get(
                "detection_prompts_path",
                "configs/prompts/error_detection_prompts.json"
            ),
        )
        
        self.game_states: dict[str, GameState] = {}
    
    def initialize_game(self, session_id: str, extra_info: dict) -> tuple[str, str]:
        """Initialize a new game, return (system_prompt, user_prompt)."""
        
        # Random mode selection
        mode = GameMode.BENIGN if random.random() < self.benign_ratio else GameMode.ERROR_INJECTION
        
        # Create game state
        state = GameState(
            mode=mode,
            turn=1,
            note_data=extra_info,
            ground_truth="CORRECT" if mode == GameMode.BENIGN else "INCORRECT",
        )
        self.game_states[session_id] = state
        
        # Get prompts from config files
        system_prompt = self.prompt_loader.get_injector_system_prompt(mode)
        user_prompt = self.prompt_loader.get_injector_user_prompt(mode, extra_info)
        
        return system_prompt, user_prompt
    
    def process_injector_response(
        self, 
        session_id: str, 
        response: str
    ) -> tuple[str, str, bool]:
        """Process Injector's response, return Assessor prompt or game end.
        
        CRITICAL: Only the generated_note text is passed to Assessor.
        All CoT reasoning, final_answer, and changes_made are stripped.
        
        Returns:
            (system_prompt, user_prompt, is_terminal)
        """
        state = self.game_states[session_id]
        state.injector_output = response
        
        # Parse the full output (for logging/analysis)
        parsed = parse_injector_output(response)
        state.injector_parsed = parsed
        
        # Extract ONLY the generated note - use the new sanitized extraction
        # This strips: <think>, final_answer, changes_made
        from scripts.self_play.cot_parser import extract_note_for_assessor
        sanitized_note = extract_note_for_assessor(response)
        
        if sanitized_note:
            state.generated_note = sanitized_note
        elif parsed.generated_note:
            # Fallback to parsed note (already stripped of CoT)
            state.generated_note = parsed.generated_note
        else:
            # No valid note extracted - will be penalized
            state.generated_note = ""
        
        # Move to turn 2 (Assessor)
        state.turn = 2
        
        # Get Assessor prompts from config files
        # The assessor sees ONLY the sanitized note - no hints about CORRECT/INCORRECT
        system_prompt = self.prompt_loader.get_assessor_system_prompt()
        user_prompt = self.prompt_loader.get_assessor_user_prompt(state.generated_note)
        
        return system_prompt, user_prompt, False
    
    def process_assessor_response(
        self,
        session_id: str,
        response: str,
    ) -> dict:
        """Process Assessor's response and complete the game.
        
        Returns:
            Game result dict for reward computation
        """
        state = self.game_states[session_id]
        state.game_complete = True
        
        return {
            "session_id": session_id,
            "mode": state.mode.value,
            "ground_truth": state.ground_truth,
            "injector_output": state.injector_output,
            "assessor_output": response,
            "generated_note": state.generated_note,
            "note_data": state.note_data,
        }
    
    def __call__(
        self,
        session_id: str,
        action: str,
        turn: int,
        extra_info: Optional[dict] = None,
    ) -> dict:
        """Main entry point called by verl during rollout.
        
        Args:
            session_id: Unique ID for this rollout
            action: The model's generated response
            turn: Current turn number (0 = init, 1 = after injector, 2 = after assessor)
            extra_info: Data from the training example
            
        Returns:
            Dict with next prompt or game results
        """
        
        if turn == 0:
            # Initialize game
            system_prompt, user_prompt = self.initialize_game(session_id, extra_info)
            return {
                "observation": {
                    "system": system_prompt,
                    "user": user_prompt,
                },
                "done": False,
            }
        
        elif turn == 1:
            # Process Injector response, get Assessor prompt
            system_prompt, user_prompt, done = self.process_injector_response(
                session_id, action
            )
            return {
                "observation": {
                    "system": system_prompt,
                    "user": user_prompt,
                },
                "done": done,
            }
        
        elif turn == 2:
            # Process Assessor response, end game
            result = self.process_assessor_response(session_id, action)
            return {
                "observation": None,
                "done": True,
                "game_result": result,
            }
        
        else:
            raise ValueError(f"Unexpected turn number: {turn}")
    
    def cleanup(self, session_id: str):
        """Clean up game state after rollout completes."""
        if session_id in self.game_states:
            del self.game_states[session_id]
