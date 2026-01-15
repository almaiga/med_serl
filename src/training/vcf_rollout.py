"""
VCF-aware rollout generator for MedSeRL online RL training.

This module provides a wrapper around vLLM generation engine that applies
Verifiable Curriculum Filtering (VCF) inline during rollouts.
"""

import sys
import os
from typing import List, Dict, Tuple, Any

# Add scripts/sft to path to import inference_utils
scripts_path = os.path.join(os.path.dirname(__file__), '../../scripts/sft')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from inference_utils import apply_vcf, FilterResult, extract_generated_note


class VCFRolloutGenerator:
    """
    Custom rollout generator for OpenRLHF that applies VCF inline.

    Integrates with OpenRLHF's vLLM rollout engine to:
    1. Generate injector outputs
    2. Apply VCF filtering
    3. Retry on rejection (max N attempts)
    4. Return only VCF-accepted samples

    Example usage:
        vcf_rollout = VCFRolloutGenerator(
            vllm_engine=vllm_engine,
            tokenizer=tokenizer,
            vcf_config={
                "min_jaccard": 0.85,
                "max_jaccard": 0.99,
                "max_word_edits": 6,
            },
            max_retries=3,
        )

        outputs, filter_results = vcf_rollout.generate_with_vcf(
            prompts=injector_prompts,
            original_notes=original_notes,
            sampling_params={"temperature": 0.7, "top_p": 0.9},
        )
    """

    def __init__(
        self,
        vllm_engine: Any,
        tokenizer: Any,
        vcf_config: Dict,
        max_retries: int = 3,
    ):
        """
        Initialize VCF rollout generator.

        Args:
            vllm_engine: vLLM generation engine (from OpenRLHF)
            tokenizer: HuggingFace tokenizer
            vcf_config: VCF configuration dict with keys:
                - min_jaccard: float (default 0.85)
                - max_jaccard: float (default 0.99)
                - max_word_edits: int (default 6)
            max_retries: Maximum number of retry attempts on VCF rejection
        """
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.vcf_config = vcf_config
        self.max_retries = max_retries

        # VCF parameters
        self.min_jaccard = vcf_config.get("min_jaccard", 0.85)
        self.max_jaccard = vcf_config.get("max_jaccard", 0.99)
        self.max_word_edits = vcf_config.get("max_word_edits", 6)

        # Statistics
        self.total_generations = 0
        self.total_rejections = 0
        self.rejection_counts_by_reason = {}

    def generate_with_vcf(
        self,
        prompts: List[str],
        original_notes: List[str],
        sampling_params: Dict,
    ) -> Tuple[List[str], List[FilterResult]]:
        """
        Generate outputs with inline VCF filtering.

        For each prompt:
        1. Generate with vLLM
        2. Extract generated note
        3. Apply VCF filtering
        4. If rejected: retry (max N times)
        5. If accepted or max retries: return output

        Args:
            prompts: Injector prompts (batch)
            original_notes: Original clinical notes (batch)
            sampling_params: vLLM generation parameters (dict with keys like
                temperature, top_p, max_tokens, etc.)

        Returns:
            Tuple of:
                - accepted_outputs: List of generated outputs (may include rejected ones if max retries)
                - filter_results: List of FilterResult for each sample
        """
        accepted_outputs = []
        filter_results = []

        for prompt, original_note in zip(prompts, original_notes):
            # Retry loop for VCF
            last_output = None
            last_filter_result = None

            for attempt in range(self.max_retries):
                # Generate with vLLM
                # Note: vLLM API may vary - adjust based on OpenRLHF integration
                try:
                    generation_output = self.vllm_engine.generate(
                        prompt,
                        sampling_params=sampling_params
                    )

                    # Extract text from vLLM output
                    # vLLM returns list of RequestOutput objects
                    if isinstance(generation_output, list):
                        output = generation_output[0].outputs[0].text
                    else:
                        output = generation_output.outputs[0].text

                except Exception as e:
                    print(f"[ERROR] vLLM generation failed: {e}")
                    # Use empty output on failure
                    output = ""

                # Extract generated note (remove <think> tags, etc.)
                generated_note = extract_generated_note(output)

                # Apply VCF
                filter_result = apply_vcf(
                    original_note,
                    generated_note,
                    min_jaccard=self.min_jaccard,
                    max_jaccard=self.max_jaccard,
                    max_word_edits=self.max_word_edits,
                )

                # Update statistics
                self.total_generations += 1
                if not filter_result.passed:
                    self.total_rejections += 1
                    reason = filter_result.reason or "unknown"
                    self.rejection_counts_by_reason[reason] = \
                        self.rejection_counts_by_reason.get(reason, 0) + 1

                # Store last attempt
                last_output = output
                last_filter_result = filter_result

                if filter_result.passed:
                    # VCF accepted - use this output
                    accepted_outputs.append(output)
                    filter_results.append(filter_result)
                    break

                # Log rejection for debugging
                if attempt < self.max_retries - 1:
                    print(
                        f"[VCF REJECT] Attempt {attempt + 1}/{self.max_retries}: "
                        f"{filter_result.reason} (Jaccard: {filter_result.score_jaccard:.3f})"
                    )

            else:
                # Max retries reached - use last attempt anyway
                print(
                    f"[VCF] Max retries ({self.max_retries}) reached. "
                    f"Using last output despite rejection: {last_filter_result.reason}"
                )
                accepted_outputs.append(last_output)
                filter_results.append(last_filter_result)

        return accepted_outputs, filter_results

    def get_vcf_statistics(self) -> Dict:
        """
        Get VCF filtering statistics.

        Returns:
            Dict with statistics:
                - total_generations: Total number of generation attempts
                - total_rejections: Total number of VCF rejections
                - rejection_rate: Fraction of rejections
                - rejection_counts_by_reason: Dict of rejection reasons and counts
        """
        rejection_rate = (
            self.total_rejections / self.total_generations
            if self.total_generations > 0
            else 0.0
        )

        return {
            "total_generations": self.total_generations,
            "total_rejections": self.total_rejections,
            "rejection_rate": rejection_rate,
            "rejection_counts_by_reason": self.rejection_counts_by_reason,
        }

    def reset_statistics(self):
        """Reset VCF statistics counters."""
        self.total_generations = 0
        self.total_rejections = 0
        self.rejection_counts_by_reason = {}
