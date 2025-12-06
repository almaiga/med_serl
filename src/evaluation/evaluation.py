"""
Evaluation Module - Model assessment on MEDEC test data.

This module provides functionality to:
- Load MEDEC test set
- Run Doctor Agent on test samples
- Compute per-error-type and overall accuracy

Requirements: 8.1, 8.2, 8.3
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.data_processor import MedicalDataProcessor
from src.training.reward_engine import calculate_reward

logger = logging.getLogger(__name__)


# The five error types from MEDEC dataset
ERROR_TYPES = [
    "Diagnosis",
    "Management",
    "Treatment",
    "Pharmacotherapy",
    "Causal Organism"
]


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        overall_accuracy: Overall classification accuracy
        per_error_type_accuracy: Accuracy for each error type
        error_detection_accuracy: Accuracy for detecting errors (binary)
        clean_detection_accuracy: Accuracy for detecting clean notes
        mean_reward: Average reward across all samples
        total_samples: Total number of samples evaluated
        error_samples: Number of error samples
        clean_samples: Number of clean samples
        confusion_matrix: Dict with TP, TN, FP, FN counts
    """
    overall_accuracy: float = 0.0
    per_error_type_accuracy: Dict[str, float] = field(default_factory=dict)
    error_detection_accuracy: float = 0.0
    clean_detection_accuracy: float = 0.0
    mean_reward: float = 0.0
    total_samples: int = 0
    error_samples: int = 0
    clean_samples: int = 0
    confusion_matrix: Dict[str, int] = field(default_factory=lambda: {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0
    })
    
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))."""
        tp = self.confusion_matrix["true_positive"]
        fp = self.confusion_matrix["false_positive"]
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))."""
        tp = self.confusion_matrix["true_positive"]
        fn = self.confusion_matrix["false_negative"]
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def false_positive_rate(self) -> float:
        """Calculate false positive rate (FP / (FP + TN))."""
        fp = self.confusion_matrix["false_positive"]
        tn = self.confusion_matrix["true_negative"]
        if fp + tn == 0:
            return 0.0
        return fp / (fp + tn)


def evaluate_model(
    model: Any,
    test_data: List[Dict],
    error_types: Optional[List[str]] = None
) -> EvaluationMetrics:
    """
    Evaluate a Doctor Agent model on test data.
    
    Runs the model on all test samples and computes accuracy metrics
    for each error type and overall.
    
    Args:
        model: DoctorAgent instance with analyze_note and parse_prediction methods
        test_data: List of test samples with 'text', 'label', and 'error_type' fields
        error_types: List of error types to evaluate (defaults to all 5 MEDEC types)
        
    Returns:
        EvaluationMetrics with overall and per-error-type accuracy
        
    Requirements: 8.1, 8.2, 8.3
    """
    if error_types is None:
        error_types = ERROR_TYPES
    
    # Initialize counters
    total_correct = 0
    total_samples = 0
    error_samples = 0
    clean_samples = 0
    
    # Per-error-type tracking
    error_type_correct: Dict[str, int] = defaultdict(int)
    error_type_total: Dict[str, int] = defaultdict(int)
    
    # Confusion matrix
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    # Reward tracking
    total_reward = 0.0
    
    logger.info(f"Evaluating model on {len(test_data)} samples...")
    
    for sample in test_data:
        note = sample.get("text", "")
        actual_label = sample.get("label", "Clean")
        actual_error_type = sample.get("error_type")
        
        if not note.strip():
            continue
        
        # Run model prediction
        # Requirements: 8.1 - Run Doctor Agent on test samples
        output = model.analyze_note(note)
        prediction = model.parse_prediction(output)
        
        predicted_label = prediction.get("predicted_label", "Clean")
        predicted_error_type = prediction.get("predicted_error_type")
        
        # Determine ground truth
        actual_has_error = actual_label == "Error"
        predicted_has_error = predicted_label == "Error"
        
        # Calculate reward
        ground_truth = {
            "has_error": actual_has_error,
            "error_type": actual_error_type
        }
        reward = calculate_reward(output, ground_truth)
        total_reward += reward
        
        # Update confusion matrix
        if actual_has_error and predicted_has_error:
            true_positive += 1
        elif not actual_has_error and not predicted_has_error:
            true_negative += 1
        elif not actual_has_error and predicted_has_error:
            false_positive += 1
        else:  # actual_has_error and not predicted_has_error
            false_negative += 1
        
        # Track overall accuracy
        total_samples += 1
        
        if actual_has_error:
            error_samples += 1
        else:
            clean_samples += 1
        
        # Check if prediction is correct
        is_correct = False
        if actual_has_error:
            # For error samples, check if error was detected
            if predicted_has_error:
                # Check if error type matches (if specified)
                if actual_error_type and predicted_error_type:
                    if _normalize_error_type(actual_error_type) == _normalize_error_type(predicted_error_type):
                        is_correct = True
                else:
                    # Credit for detecting error even without type match
                    is_correct = True
        else:
            # For clean samples, check if predicted clean
            if not predicted_has_error:
                is_correct = True
        
        if is_correct:
            total_correct += 1
        
        # Requirements: 8.2 - Compute accuracy for each error type
        if actual_error_type:
            normalized_type = _normalize_error_type(actual_error_type)
            error_type_total[normalized_type] += 1
            
            if predicted_has_error and predicted_error_type:
                if _normalize_error_type(predicted_error_type) == normalized_type:
                    error_type_correct[normalized_type] += 1
    
    # Requirements: 8.3 - Compute overall classification accuracy
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Compute per-error-type accuracy
    per_error_type_accuracy = {}
    for error_type in error_types:
        normalized = _normalize_error_type(error_type)
        total = error_type_total.get(normalized, 0)
        correct = error_type_correct.get(normalized, 0)
        per_error_type_accuracy[error_type] = correct / total if total > 0 else 0.0
    
    # Compute error/clean detection accuracy
    error_detection_accuracy = true_positive / error_samples if error_samples > 0 else 0.0
    clean_detection_accuracy = true_negative / clean_samples if clean_samples > 0 else 0.0
    
    # Mean reward
    mean_reward = total_reward / total_samples if total_samples > 0 else 0.0
    
    metrics = EvaluationMetrics(
        overall_accuracy=overall_accuracy,
        per_error_type_accuracy=per_error_type_accuracy,
        error_detection_accuracy=error_detection_accuracy,
        clean_detection_accuracy=clean_detection_accuracy,
        mean_reward=mean_reward,
        total_samples=total_samples,
        error_samples=error_samples,
        clean_samples=clean_samples,
        confusion_matrix={
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative
        }
    )
    
    logger.info(f"Evaluation complete: accuracy={overall_accuracy:.3f}, "
                f"mean_reward={mean_reward:.3f}")
    
    return metrics


def compute_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> Dict[str, float]:
    """
    Compute evaluation metrics from predictions and ground truth.
    
    This is a lower-level function that takes pre-computed predictions
    and ground truth labels.
    
    Args:
        predictions: List of prediction dicts with 'predicted_label' and 
                    'predicted_error_type' fields
        ground_truth: List of ground truth dicts with 'has_error' and 
                     'error_type' fields
                     
    Returns:
        Dictionary of metric names to values
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) "
            "must have same length"
        )
    
    total = len(predictions)
    if total == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for pred, gt in zip(predictions, ground_truth):
        pred_has_error = pred.get("predicted_label") == "Error"
        actual_has_error = gt.get("has_error", False)
        
        if pred_has_error == actual_has_error:
            correct += 1
        
        if actual_has_error and pred_has_error:
            tp += 1
        elif not actual_has_error and not pred_has_error:
            tn += 1
        elif not actual_has_error and pred_has_error:
            fp += 1
        else:
            fn += 1
    
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }


def _normalize_error_type(error_type: str) -> str:
    """
    Normalize error type string for comparison.
    
    Handles variations like 'causalOrganism' -> 'causal organism'.
    """
    if not error_type:
        return ""
    
    # Convert camelCase to space-separated
    import re
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', error_type)
    return result.lower().strip()


def load_test_data(
    data_path: str = "data_raw/MEDEC",
    subsets: Optional[List[str]] = None
) -> List[Dict]:
    """
    Load MEDEC test data for evaluation.
    
    Args:
        data_path: Path to MEDEC data directory
        subsets: List of subsets to load (default: ['MS', 'UW'])
        
    Returns:
        List of test samples with 'text', 'label', and 'error_type' fields
        
    Requirements: 8.1
    """
    # Use the factory method to load only test data
    processor = MedicalDataProcessor.load_test_data(
        data_path=data_path,
        subsets=subsets
    )
    
    # Combine error and clean pools into test samples
    test_samples = []
    
    # Add error samples
    for entry in processor.error_pool:
        test_samples.append({
            "text": entry.get("text", ""),
            "text_id": entry.get("text_id", ""),
            "label": "Error",
            "error_type": entry.get("error_type"),
            "subset": entry.get("subset", "")
        })
    
    # Add clean samples
    for entry in processor.clean_pool:
        test_samples.append({
            "text": entry.get("text", ""),
            "text_id": entry.get("text_id", ""),
            "label": "Clean",
            "error_type": None,
            "subset": entry.get("subset", "")
        })
    
    logger.info(f"Loaded {len(test_samples)} test samples "
                f"({len(processor.error_pool)} errors, {len(processor.clean_pool)} clean)")
    
    return test_samples


def run_evaluation(
    model: Any,
    data_path: str = "data_raw/MEDEC",
    subsets: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> EvaluationMetrics:
    """
    Run full evaluation pipeline on MEDEC test data.
    
    Convenience function that loads test data and evaluates the model.
    
    Args:
        model: DoctorAgent instance
        data_path: Path to MEDEC data
        subsets: Subsets to evaluate on
        max_samples: Maximum samples to evaluate (for quick testing)
        
    Returns:
        EvaluationMetrics with full results
        
    Requirements: 8.1, 8.2, 8.3
    """
    # Load test data
    test_data = load_test_data(data_path, subsets)
    
    # Optionally limit samples
    if max_samples and len(test_data) > max_samples:
        import random
        test_data = random.sample(test_data, max_samples)
        logger.info(f"Limited evaluation to {max_samples} samples")
    
    # Run evaluation
    return evaluate_model(model, test_data)


def print_evaluation_report(metrics: EvaluationMetrics) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: EvaluationMetrics to report
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Total Samples: {metrics.total_samples}")
    print(f"  Error Samples: {metrics.error_samples}")
    print(f"  Clean Samples: {metrics.clean_samples}")
    print(f"  Overall Accuracy: {metrics.overall_accuracy:.3f}")
    print(f"  Mean Reward: {metrics.mean_reward:.3f}")
    
    print(f"\nDetection Metrics:")
    print(f"  Error Detection Accuracy: {metrics.error_detection_accuracy:.3f}")
    print(f"  Clean Detection Accuracy: {metrics.clean_detection_accuracy:.3f}")
    print(f"  Precision: {metrics.precision():.3f}")
    print(f"  Recall: {metrics.recall():.3f}")
    print(f"  F1 Score: {metrics.f1_score():.3f}")
    print(f"  False Positive Rate: {metrics.false_positive_rate():.3f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics.confusion_matrix
    print(f"  True Positive:  {cm['true_positive']}")
    print(f"  True Negative:  {cm['true_negative']}")
    print(f"  False Positive: {cm['false_positive']}")
    print(f"  False Negative: {cm['false_negative']}")
    
    print(f"\nPer-Error-Type Accuracy:")
    for error_type, accuracy in metrics.per_error_type_accuracy.items():
        print(f"  {error_type}: {accuracy:.3f}")
    
    print("=" * 60)
