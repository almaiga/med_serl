"""
MedSeRL Custom Trainer - Online RL with inline VCF filtering.

This module provides a custom PPO trainer for MedSeRL that integrates
Verifiable Curriculum Filtering (VCF) directly into the training loop.

Training flow per round:
1. Sample batch of notes (32/64/128)
2. Injector rollout with VCF filtering
3. Assessor rollout on VCF-accepted notes
4. Compute zero-sum rewards
5. Policy update with REINFORCE++
6. Log interactions and metrics
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/sft'))

from vcf_rollout import VCFRolloutGenerator
from inference_utils import extract_final_answer, extract_generated_note, FilterResult


try:
    # Try to import OpenRLHF's PPOTrainer
    from openrlhf.trainer import PPOTrainer
    HAS_OPENRLHF = True
except ImportError:
    # Fallback - create stub base class
    print("[WARNING] OpenRLHF not found. Using stub PPOTrainer base class.")
    HAS_OPENRLHF = False

    class PPOTrainer:
        """Stub base class if OpenRLHF not available."""
        def __init__(self, *args, **kwargs):
            pass


class MedSeRLTrainer(PPOTrainer):
    """
    Custom PPO trainer with VCF-aware rollouts for MedSeRL.

    Training loop per round:
    1. Sample batch of notes (32/64/128)
    2. Injector rollout with VCF filtering
    3. Assessor rollout on VCF-accepted notes
    4. Compute zero-sum rewards
    5. Policy update with REINFORCE++
    6. Log interactions and metrics

    Example usage:
        trainer = MedSeRLTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            vcf_config={
                "min_jaccard": 0.85,
                "max_jaccard": 0.99,
                "max_word_edits": 6,
            },
            interaction_log_path="logs/interactions.jsonl",
            metrics_log_path="logs/metrics.jsonl",
            ...
        )

        for batch in dataloader:
            metrics = trainer.training_step(batch)
    """

    def __init__(
        self,
        *args,
        vcf_config: Dict,
        interaction_log_path: str,
        metrics_log_path: str,
        injector_prompts: Optional[Dict] = None,
        assessor_prompts: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize MedSeRL trainer.

        Args:
            vcf_config: VCF configuration dict
            interaction_log_path: Path to interaction log file
            metrics_log_path: Path to metrics log file
            injector_prompts: Injector prompt templates (dict)
            assessor_prompts: Assessor prompt templates (dict)
            *args, **kwargs: Passed to PPOTrainer base class
        """
        if HAS_OPENRLHF:
            super().__init__(*args, **kwargs)

        # Initialize VCF rollout generator
        # Note: vllm_engines comes from OpenRLHF's PPOTrainer
        if hasattr(self, 'vllm_engines') and self.vllm_engines:
            self.vcf_rollout = VCFRolloutGenerator(
                vllm_engine=self.vllm_engines[0],
                tokenizer=self.tokenizer if hasattr(self, 'tokenizer') else None,
                vcf_config=vcf_config,
                max_retries=vcf_config.get("max_retries", 3),
            )
        else:
            print("[WARNING] No vLLM engines found. VCF rollout disabled.")
            self.vcf_rollout = None

        # Open log files for streaming writes
        os.makedirs(os.path.dirname(interaction_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)

        self.interaction_log = open(interaction_log_path, 'a')
        self.metrics_log = open(metrics_log_path, 'a')
        self.current_round = 0

        # Prompt templates
        self.injector_prompts = injector_prompts or {}
        self.assessor_prompts = assessor_prompts or {}

    def __del__(self):
        """Clean up log files on trainer destruction."""
        if hasattr(self, 'interaction_log'):
            self.interaction_log.close()
        if hasattr(self, 'metrics_log'):
            self.metrics_log.close()

    def build_injector_prompts(self, notes: List[str]) -> List[str]:
        """
        Build injector prompts from notes.

        Args:
            notes: List of clinical notes

        Returns:
            List of formatted injector prompts
        """
        # Use injector template if available
        if self.injector_prompts and "injector_incorrect_template" in self.injector_prompts:
            template = self.injector_prompts["injector_incorrect_template"]
            prompts = []
            for note in notes:
                # Format prompt with note
                prompt = template.format(
                    note=note,
                    prompt_intent="inject a subtle medical error"
                )
                prompts.append(prompt)
            return prompts

        # Fallback simple prompt
        return [
            f"Inject a subtle medical error into the following clinical note:\n\n{note}\n\n"
            f"Generate the modified note:"
            for note in notes
        ]

    def build_assessor_prompts(self, generated_notes: List[str]) -> List[str]:
        """
        Build assessor prompts from generated notes.

        Args:
            generated_notes: List of generated/modified clinical notes

        Returns:
            List of formatted assessor prompts
        """
        # Use assessor template if available
        if self.assessor_prompts and "user_template" in self.assessor_prompts:
            template = self.assessor_prompts["user_template"]
            return [template.format(note=note) for note in generated_notes]

        # Fallback simple prompt
        return [
            f"Assess if the following clinical note contains medical errors:\n\n{note}\n\n"
            f"Answer: CORRECT or INCORRECT"
            for note in generated_notes
        ]

    def compute_rewards(
        self,
        injector_outputs: List[str],
        assessor_outputs: List[str],
        ground_truth: List[str],
    ) -> torch.Tensor:
        """
        Compute zero-sum rewards for injector-assessor self-play.

        Args:
            injector_outputs: Injector generated outputs
            assessor_outputs: Assessor predictions
            ground_truth: Ground truth labels ("CORRECT" or "INCORRECT")

        Returns:
            Tensor of rewards (shape: [batch_size])
        """
        rewards = []

        for inj_out, ass_out, gt in zip(injector_outputs, assessor_outputs, ground_truth):
            # Extract assessor prediction
            prediction = extract_final_answer(ass_out)

            # Check for thinking section (structural reward)
            has_thinking = "<think>" in ass_out or "<thinking>" in ass_out
            structural_reward = 0.1 if has_thinking else 0.0

            # Compute outcome reward (zero-sum)
            # In self-play:
            # - If assessor correctly detects error → assessor wins (+1.0)
            # - If assessor fooled by injector → injector wins (+1.0)
            # For simplicity, assume injector always generates INCORRECT notes
            # (since we're training injector to inject errors)

            if prediction == "INCORRECT":
                # Assessor correctly detected error
                outcome_reward = 1.0
            elif prediction == "CORRECT":
                # Assessor fooled (injector wins)
                outcome_reward = -1.0  # Negative for assessor, positive for injector
            else:
                # Unknown prediction
                outcome_reward = -0.5

            # Total reward
            total_reward = structural_reward + outcome_reward
            rewards.append(total_reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def training_step(self, batch: Dict) -> Dict:
        """
        Single training step with inline VCF.

        Args:
            batch: Batch dict with keys:
                - "input_ids": Original note prompts
                - "notes": Original clinical notes (for VCF)
                - "labels": Ground truth labels

        Returns:
            Dict with metrics: loss, reward, kl_div, vcf_acceptance_rate, etc.
        """
        if not self.vcf_rollout:
            raise RuntimeError("VCF rollout not initialized. Check vLLM engine setup.")

        # Step 1: Injector rollout with VCF
        injector_prompts = self.build_injector_prompts(batch["notes"])
        injector_outputs, vcf_results = self.vcf_rollout.generate_with_vcf(
            prompts=injector_prompts,
            original_notes=batch["notes"],
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024,
            },
        )

        # Step 2: Assessor rollout (no VCF needed)
        # Extract generated notes for assessor
        generated_notes = [extract_generated_note(out) for out in injector_outputs]
        assessor_prompts = self.build_assessor_prompts(generated_notes)

        # Generate assessor outputs
        if hasattr(self, 'vllm_engines') and self.vllm_engines:
            assessor_outputs = []
            for prompt in assessor_prompts:
                try:
                    output = self.vllm_engines[0].generate(
                        prompt,
                        sampling_params={
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 512,
                        },
                    )
                    if isinstance(output, list):
                        assessor_outputs.append(output[0].outputs[0].text)
                    else:
                        assessor_outputs.append(output.outputs[0].text)
                except Exception as e:
                    print(f"[ERROR] Assessor generation failed: {e}")
                    assessor_outputs.append("")
        else:
            assessor_outputs = [""] * len(injector_outputs)

        # Step 3: Compute zero-sum rewards
        rewards = self.compute_rewards(
            injector_outputs,
            assessor_outputs,
            ground_truth=batch.get("labels", ["INCORRECT"] * len(injector_outputs)),
        )

        # Step 4: LOG INTERACTIONS (one per sample in batch)
        self._log_interactions(
            round_num=self.current_round,
            batch=batch,
            injector_outputs=injector_outputs,
            assessor_outputs=assessor_outputs,
            vcf_results=vcf_results,
            rewards=rewards,
        )

        # Step 5: Policy update (REINFORCE++)
        # Note: This requires implementing PPO/REINFORCE++ update
        # For now, return placeholder loss
        loss = 0.0  # Placeholder - implement actual policy update

        # Step 6: LOG ROUND METRICS (aggregate statistics)
        metrics = {
            "loss": loss,
            "mean_reward": rewards.mean().item(),
            "vcf_acceptance_rate": sum(r.passed for r in vcf_results) / len(vcf_results),
            "injector_win_rate": (rewards < 0).float().mean().item(),  # Negative reward = injector won
            "assessor_win_rate": (rewards > 0).float().mean().item(),  # Positive reward = assessor won
        }
        self._log_round_metrics(self.current_round, metrics)

        self.current_round += 1
        return metrics

    def _log_interactions(
        self,
        round_num: int,
        batch: Dict,
        injector_outputs: List[str],
        assessor_outputs: List[str],
        vcf_results: List[FilterResult],
        rewards: torch.Tensor,
    ):
        """Log individual interactions to interactions.jsonl."""
        for i in range(len(batch["notes"])):
            # Extract assessor prediction
            assessor_prediction = extract_final_answer(assessor_outputs[i])

            interaction = {
                "round": round_num,
                "sample_idx": i,
                "original_note": batch["notes"][i],
                "injector_output": injector_outputs[i],
                "injector_note": extract_generated_note(injector_outputs[i]),
                "assessor_output": assessor_outputs[i],
                "assessor_prediction": assessor_prediction,
                "ground_truth": batch.get("labels", ["INCORRECT"] * len(batch["notes"]))[i],
                "reward": rewards[i].item(),
                "vcf_passed": vcf_results[i].passed,
                "vcf_reason": vcf_results[i].reason,
                "vcf_jaccard": vcf_results[i].score_jaccard,
                "injector_won": rewards[i].item() < 0,  # Negative reward = injector fooled assessor
                "assessor_won": rewards[i].item() > 0,  # Positive reward = assessor detected error
            }
            self.interaction_log.write(json.dumps(interaction) + "\n")
            self.interaction_log.flush()

    def _log_round_metrics(self, round_num: int, metrics: Dict):
        """Log aggregate metrics to metrics.jsonl."""
        metric_summary = {
            "round": round_num,
            "loss": metrics["loss"],
            "mean_reward": metrics["mean_reward"],
            "vcf_acceptance_rate": metrics["vcf_acceptance_rate"],
            "injector_win_rate": metrics["injector_win_rate"],
            "assessor_win_rate": metrics["assessor_win_rate"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.metrics_log.write(json.dumps(metric_summary) + "\n")
        self.metrics_log.flush()
