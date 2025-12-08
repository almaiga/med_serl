"""
Interaction Logger for MedSeRL Training

Logs model interactions (prompts, responses, rewards) for debugging,
analysis, and potential manual correction of self-play behavior.

Usage:
    logger = InteractionLogger("outputs/interactions")
    logger.log_interaction(
        episode=1,
        note=note_text,
        quadrant="synthetic_injection",
        ground_truth={"has_error": True, "error_type": "Pharmacotherapy"},
        model_output="<thinking>...</thinking><verdict>Error</verdict>",
        reward=1.0,
        metadata={"temperature": 0.7}
    )
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Interaction:
    """Single model interaction record."""
    timestamp: str
    episode: int
    batch_idx: int
    
    # Input
    note: str
    quadrant: str  # augmented_gt, augmented_safe, synthetic_decoy, synthetic_injection
    ground_truth: Dict[str, Any]
    
    # Output
    model_output: str
    parsed_verdict: Optional[str]
    parsed_reasoning: Optional[str]
    
    # Reward
    reward: float
    reward_breakdown: Optional[Dict[str, float]]
    
    # Classification result
    correct: bool
    error_type: Optional[str]  # false_positive, false_negative, correct
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


class InteractionLogger:
    """
    Logs training interactions to JSONL files for analysis.
    
    Creates one file per training session with all interactions.
    Also maintains a separate file for "interesting" cases (failures, edge cases).
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/interactions",
        session_name: Optional[str] = None,
        log_all: bool = True,
        log_failures_only: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session naming
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        
        # File paths
        self.all_interactions_path = self.output_dir / f"{session_name}_all.jsonl"
        self.failures_path = self.output_dir / f"{session_name}_failures.jsonl"
        self.summary_path = self.output_dir / f"{session_name}_summary.json"
        
        self.log_all = log_all
        self.log_failures_only = log_failures_only
        
        # Running statistics
        self.stats = {
            "total": 0,
            "correct": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "by_quadrant": {},
            "by_error_type": {},
            "reward_sum": 0.0,
        }
        
        logger.info(f"InteractionLogger initialized: {self.all_interactions_path}")
    
    def log_interaction(
        self,
        episode: int,
        batch_idx: int,
        note: str,
        quadrant: str,
        ground_truth: Dict[str, Any],
        model_output: str,
        reward: float,
        reward_breakdown: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Interaction:
        """Log a single interaction."""
        
        # Parse model output
        parsed_verdict, parsed_reasoning = self._parse_output(model_output)
        
        # Determine correctness
        correct, error_type = self._evaluate_correctness(
            parsed_verdict, ground_truth
        )
        
        interaction = Interaction(
            timestamp=datetime.now().isoformat(),
            episode=episode,
            batch_idx=batch_idx,
            note=note[:500] + "..." if len(note) > 500 else note,  # Truncate long notes
            quadrant=quadrant,
            ground_truth=ground_truth,
            model_output=model_output,
            parsed_verdict=parsed_verdict,
            parsed_reasoning=parsed_reasoning[:300] if parsed_reasoning else None,
            reward=reward,
            reward_breakdown=reward_breakdown,
            correct=correct,
            error_type=error_type,
            metadata=metadata,
        )
        
        # Update stats
        self._update_stats(interaction)
        
        # Write to files
        if self.log_all and not self.log_failures_only:
            self._append_jsonl(self.all_interactions_path, interaction)
        
        if not correct:
            self._append_jsonl(self.failures_path, interaction)
        
        return interaction
    
    def log_batch(
        self,
        episode: int,
        notes: List[str],
        quadrants: List[str],
        ground_truths: List[Dict[str, Any]],
        model_outputs: List[str],
        rewards: List[float],
        reward_breakdowns: Optional[List[Dict[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Interaction]:
        """Log a batch of interactions."""
        interactions = []
        
        for i, (note, quad, gt, output, reward) in enumerate(
            zip(notes, quadrants, ground_truths, model_outputs, rewards)
        ):
            breakdown = reward_breakdowns[i] if reward_breakdowns else None
            interaction = self.log_interaction(
                episode=episode,
                batch_idx=i,
                note=note,
                quadrant=quad,
                ground_truth=gt,
                model_output=output,
                reward=reward,
                reward_breakdown=breakdown,
                metadata=metadata,
            )
            interactions.append(interaction)
        
        return interactions
    
    def _parse_output(self, output: str) -> tuple[Optional[str], Optional[str]]:
        """Extract answer and reasoning from model output."""
        import re

        # Extract thinking (supports both <think> and <thinking>)
        thinking_match = re.search(
            r'<think(?:ing)?>(.*?)</think(?:ing)?>', output, re.DOTALL
        )
        reasoning = thinking_match.group(1).strip() if thinking_match else None

        # Extract answer - try multiple formats
        # 1. Try <answer> tags first (preferred)
        answer_match = re.search(
            r'<answer>\s*(CORRECT|INCORRECT)\s*</answer>', output, re.IGNORECASE
        )
        if answer_match:
            return answer_match.group(1).upper(), reasoning

        # 2. Try <verdict> tags (legacy format)
        verdict_match = re.search(
            r'<verdict>\s*([^<]+)\s*</verdict>', output, re.IGNORECASE
        )
        if verdict_match:
            verdict_content = verdict_match.group(1).lower().strip()
            if any(p in verdict_content for p in [
                'no error', 'correct', 'accurate', 'no obvious error'
            ]):
                return 'CORRECT', reasoning
            elif 'error' in verdict_content or 'incorrect' in verdict_content:
                return 'INCORRECT', reasoning

        # 3. Look for clear indicators in text
        output_lower = output.lower()
        if any(p in output_lower for p in [
            'no errors detected', 'no errors found', 'no error',
            'appears correct', 'is correct'
        ]):
            return 'CORRECT', reasoning
        elif any(p in output_lower for p in [
            'error detected', 'error found', 'contains error',
            'found an error', 'identified an error'
        ]):
            return 'INCORRECT', reasoning

        return None, reasoning
    
    def _evaluate_correctness(
        self, verdict: Optional[str], ground_truth: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Determine if prediction was correct."""
        if verdict is None:
            return False, "parse_error"

        has_error = ground_truth.get("has_error", False)
        # verdict is now "CORRECT" or "INCORRECT"
        predicted_error = (verdict == "INCORRECT")

        if has_error and predicted_error:
            return True, "true_positive"
        elif not has_error and not predicted_error:
            return True, "true_negative"
        elif not has_error and predicted_error:
            return False, "false_positive"
        else:
            return False, "false_negative"
    
    def _update_stats(self, interaction: Interaction):
        """Update running statistics."""
        self.stats["total"] += 1
        self.stats["reward_sum"] += interaction.reward
        
        if interaction.correct:
            self.stats["correct"] += 1
        elif interaction.error_type == "false_positive":
            self.stats["false_positives"] += 1
        elif interaction.error_type == "false_negative":
            self.stats["false_negatives"] += 1
        
        # By quadrant
        quad = interaction.quadrant
        if quad not in self.stats["by_quadrant"]:
            self.stats["by_quadrant"][quad] = {"total": 0, "correct": 0}
        self.stats["by_quadrant"][quad]["total"] += 1
        if interaction.correct:
            self.stats["by_quadrant"][quad]["correct"] += 1
    
    def _append_jsonl(self, path: Path, interaction: Interaction):
        """Append interaction to JSONL file."""
        with open(path, "a") as f:
            f.write(json.dumps(asdict(interaction)) + "\n")
    
    def save_summary(self):
        """Save summary statistics."""
        summary = {
            **self.stats,
            "accuracy": self.stats["correct"] / max(self.stats["total"], 1),
            "mean_reward": self.stats["reward_sum"] / max(self.stats["total"], 1),
            "quadrant_accuracy": {
                k: v["correct"] / max(v["total"], 1)
                for k, v in self.stats["by_quadrant"].items()
            },
        }
        
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved: {self.summary_path}")
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            "accuracy": self.stats["correct"] / max(self.stats["total"], 1),
            "mean_reward": self.stats["reward_sum"] / max(self.stats["total"], 1),
        }
    
    def close(self):
        """Finalize logging session."""
        self.save_summary()
        logger.info(
            f"Session complete: {self.stats['total']} interactions, "
            f"{self.stats['correct']/max(self.stats['total'],1):.1%} accuracy"
        )
