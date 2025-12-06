"""
MedSeRL Reward Server for OpenRLHF Integration.

This module provides a FastAPI-based HTTP server that wraps the
calculate_reward function as a remote reward model for OpenRLHF.

The server accepts POST requests with model outputs and ground truth,
returning computed rewards.

Requirements: 9.4
"""

import argparse
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.training.reward_engine import (
    calculate_reward,
    calculate_reward_with_metadata,
    RewardMetadata
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

if HAS_FASTAPI:
    class RewardRequest(BaseModel):
        """Request model for reward calculation."""
        queries: List[str]  # Model outputs
        prompts: List[str]  # Original prompts (optional context)
        labels: Optional[List[str]] = None  # Ground truth labels

    class RewardResponse(BaseModel):
        """Response model for reward calculation."""
        rewards: List[float]

    class DetailedRewardResponse(BaseModel):
        """Detailed response with reward breakdown."""
        rewards: List[float]
        metadata: List[Dict[str, Any]]


# =============================================================================
# Ground Truth Manager
# =============================================================================

@dataclass
class GroundTruthEntry:
    """A single ground truth entry."""
    has_error: bool
    error_type: Optional[str] = None
    source: Optional[str] = None


class GroundTruthManager:
    """
    Manages ground truth labels for reward calculation.

    In MedSeRL, ground truth comes from the Scribe Agent's transformation
    metadata. This manager stores and retrieves ground truth based on
    prompt/note identifiers.
    """

    def __init__(self):
        self._cache: Dict[str, GroundTruthEntry] = {}
        self._default_ground_truth = GroundTruthEntry(has_error=False)

    def register(
        self,
        prompt_id: str,
        has_error: bool,
        error_type: Optional[str] = None,
        source: Optional[str] = None
    ) -> None:
        """Register ground truth for a prompt."""
        self._cache[prompt_id] = GroundTruthEntry(
            has_error=has_error,
            error_type=error_type,
            source=source
        )

    def get(self, prompt_id: str) -> Dict[str, Any]:
        """Get ground truth for a prompt."""
        entry = self._cache.get(prompt_id, self._default_ground_truth)
        return {
            "has_error": entry.has_error,
            "error_type": entry.error_type
        }

    def clear(self) -> None:
        """Clear all cached ground truth."""
        self._cache.clear()

    def register_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Register ground truth for a batch of samples."""
        for item in batch:
            prompt_id = item.get("prompt_id", item.get("scribe_prompt", ""))
            ground_truth = item.get("ground_truth", item.get("meta", {}))
            self.register(
                prompt_id=prompt_id,
                has_error=ground_truth.get("has_error", False),
                error_type=ground_truth.get("error_type"),
                source=ground_truth.get("source")
            )


# =============================================================================
# Reward Function Wrapper
# =============================================================================

def medserl_reward_func(
    queries: List[str],
    prompts: List[str],
    labels: Optional[List[str]] = None,
    ground_truth_manager: Optional[GroundTruthManager] = None
) -> List[float]:
    """
    MedSeRL reward function compatible with OpenRLHF interface.

    This function wraps calculate_reward to match the interface expected
    by OpenRLHF's remote reward model.

    Args:
        queries: List of model outputs (Doctor Agent responses)
        prompts: List of original prompts (clinical notes)
        labels: Optional list of ground truth labels
        ground_truth_manager: Optional manager for ground truth lookup

    Returns:
        List of reward values

    Requirements: 9.4
    """
    rewards = []

    for i, query in enumerate(queries):
        # Determine ground truth
        if labels and i < len(labels):
            # Parse label string to ground truth dict
            label = labels[i]
            ground_truth = _parse_label_to_ground_truth(label)
        elif ground_truth_manager and i < len(prompts):
            # Look up ground truth from manager
            prompt = prompts[i]
            ground_truth = ground_truth_manager.get(prompt)
        else:
            # Default: assume clean note
            ground_truth = {"has_error": False}

        # Calculate reward
        reward = calculate_reward(query, ground_truth)
        rewards.append(reward)

    return rewards


def _parse_label_to_ground_truth(label: str) -> Dict[str, Any]:
    """
    Parse a label string to ground truth dictionary.

    Supports formats:
    - "Error: Diagnosis" -> {"has_error": True, "error_type": "Diagnosis"}
    - "Clean" or "No Error" -> {"has_error": False}
    - JSON string -> parsed dict

    Args:
        label: Label string

    Returns:
        Ground truth dictionary
    """
    if not label:
        return {"has_error": False}

    label_lower = label.lower().strip()

    # Check for clean/no error
    if label_lower in ["clean", "no error", "no clinical error"]:
        return {"has_error": False}

    # Check for error format
    if label_lower.startswith("error"):
        # Try to extract error type
        if ":" in label:
            error_type = label.split(":", 1)[1].strip()
            return {"has_error": True, "error_type": error_type}
        return {"has_error": True}

    # Try JSON parsing
    try:
        import json
        return json.loads(label)
    except (json.JSONDecodeError, TypeError):
        pass

    # Default: treat as error type
    return {"has_error": True, "error_type": label}


# =============================================================================
# FastAPI Server
# =============================================================================

def create_app(ground_truth_manager: Optional[GroundTruthManager] = None):
    """
    Create FastAPI application for reward server.

    Args:
        ground_truth_manager: Optional ground truth manager

    Returns:
        FastAPI application instance
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required for reward server. "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="MedSeRL Reward Server",
        description="Remote reward model for MedSeRL training with OpenRLHF",
        version="1.0.0"
    )

    gt_manager = ground_truth_manager or GroundTruthManager()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "medserl-reward"}

    @app.post("/reward", response_model=RewardResponse)
    async def compute_reward(request: RewardRequest):
        """
        Compute rewards for model outputs.

        This endpoint is called by OpenRLHF during training to get
        rewards for generated responses.

        Requirements: 9.4
        """
        try:
            rewards = medserl_reward_func(
                queries=request.queries,
                prompts=request.prompts,
                labels=request.labels,
                ground_truth_manager=gt_manager
            )
            return RewardResponse(rewards=rewards)
        except Exception as e:
            logger.error(f"Error computing rewards: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reward/detailed", response_model=DetailedRewardResponse)
    async def compute_reward_detailed(request: RewardRequest):
        """
        Compute rewards with detailed metadata breakdown.

        Returns reward components (structural, outcome, penalties)
        for debugging and analysis.
        """
        try:
            rewards = []
            metadata = []

            for i, query in enumerate(request.queries):
                # Get ground truth
                if request.labels and i < len(request.labels):
                    ground_truth = _parse_label_to_ground_truth(request.labels[i])
                elif i < len(request.prompts):
                    ground_truth = gt_manager.get(request.prompts[i])
                else:
                    ground_truth = {"has_error": False}

                # Calculate with metadata
                result = calculate_reward_with_metadata(query, ground_truth)
                rewards.append(result.total_reward)
                metadata.append({
                    "structural_reward": result.structural_reward,
                    "outcome_reward": result.outcome_reward,
                    "total_reward": result.total_reward,
                    "correct_classification": result.correct_classification,
                    "false_positive": result.false_positive,
                    "false_negative": result.false_negative
                })

            return DetailedRewardResponse(rewards=rewards, metadata=metadata)
        except Exception as e:
            logger.error(f"Error computing detailed rewards: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ground_truth/register")
    async def register_ground_truth(batch: List[Dict[str, Any]]):
        """
        Register ground truth for a batch of samples.

        Called by the training loop to register ground truth from
        Scribe Agent transformations.
        """
        try:
            gt_manager.register_batch(batch)
            return {"status": "ok", "registered": len(batch)}
        except Exception as e:
            logger.error(f"Error registering ground truth: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ground_truth/clear")
    async def clear_ground_truth():
        """Clear all cached ground truth."""
        gt_manager.clear()
        return {"status": "ok"}

    return app


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the reward server."""
    parser = argparse.ArgumentParser(
        description="MedSeRL Reward Server for OpenRLHF"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on"
    )
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_raw/MEDEC",
        help="Path to MEDEC dataset (for ground truth)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    if not HAS_FASTAPI:
        print("Error: FastAPI is required. Install with: pip install fastapi uvicorn")
        return 1

    # Create and run app
    app = create_app()

    logger.info(f"Starting MedSeRL Reward Server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
