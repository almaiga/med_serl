"""
Tests for reward engine module.

Tests the deterministic reward calculation for MedSeRL Doctor Agent predictions.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.training.reward_engine import (
    calculate_reward,
    calculate_reward_with_metadata,
    has_thinking_section,
    parse_verdict,
    STRUCTURAL_REWARD,
    CORRECT_CLASSIFICATION_REWARD,
    FALSE_NEGATIVE_PENALTY,
    FALSE_POSITIVE_PENALTY,
    RewardMetadata
)


# =============================================================================
# Hypothesis Strategies for Property-Based Testing
# =============================================================================

# Valid error types from MEDEC
ERROR_TYPES = ["Diagnosis", "Management", "Treatment", "Pharmacotherapy", "Causal Organism"]

# Strategy for generating thinking section content
thinking_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
    min_size=1,
    max_size=200
).filter(lambda x: x.strip())

# Strategy for generating error types
error_type_strategy = st.sampled_from(ERROR_TYPES)

# Strategy for generating model outputs WITH thinking section
@st.composite
def output_with_thinking(draw):
    """Generate model output that contains a <thinking> section."""
    thinking = draw(thinking_content_strategy)
    has_error = draw(st.booleans())
    
    if has_error:
        error_type = draw(error_type_strategy)
        verdict = f"Error: {error_type}"
    else:
        verdict = "No Clinical Error"
    
    return f"<thinking>{thinking}</thinking><verdict>{verdict}</verdict>"

# Strategy for generating model outputs WITHOUT thinking section
@st.composite
def output_without_thinking(draw):
    """Generate model output that does NOT contain a <thinking> section."""
    has_error = draw(st.booleans())
    
    if has_error:
        error_type = draw(error_type_strategy)
        verdict = f"Error: {error_type}"
    else:
        verdict = "No Clinical Error"
    
    return f"<verdict>{verdict}</verdict>"

# Strategy for generating ground truth for error notes
@st.composite
def error_ground_truth(draw):
    """Generate ground truth for a note that contains an error."""
    error_type = draw(error_type_strategy)
    return {"has_error": True, "error_type": error_type}

# Strategy for generating ground truth for clean notes
def clean_ground_truth():
    """Generate ground truth for a clean note."""
    return st.just({"has_error": False})

# Strategy for correct error detection output (matches ground truth)
@st.composite
def correct_error_detection_output(draw, error_type: str):
    """Generate output that correctly identifies the given error type."""
    thinking = draw(thinking_content_strategy)
    return f"<thinking>{thinking}</thinking><verdict>Error: {error_type}</verdict>"

# Strategy for false negative output (says no error when there is one)
@st.composite
def false_negative_output(draw):
    """Generate output that says 'No Clinical Error' (for false negative scenarios)."""
    thinking = draw(thinking_content_strategy)
    return f"<thinking>{thinking}</thinking><verdict>No Clinical Error</verdict>"

# Strategy for correct clean classification output
@st.composite
def correct_clean_output(draw):
    """Generate output that correctly identifies a clean note."""
    thinking = draw(thinking_content_strategy)
    return f"<thinking>{thinking}</thinking><verdict>No Clinical Error</verdict>"

# Strategy for false positive output (says error when there is none)
@st.composite
def false_positive_output(draw):
    """Generate output that incorrectly identifies an error on a clean note."""
    thinking = draw(thinking_content_strategy)
    error_type = draw(error_type_strategy)
    return f"<thinking>{thinking}</thinking><verdict>Error: {error_type}</verdict>"


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestPropertyBasedRewardCalculation:
    """
    Property-based tests for reward calculation.
    
    **Feature: medserl-adaptation, Property 12-17: Reward Calculation Properties**
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
    """

    @settings(max_examples=100)
    @given(output=output_with_thinking())
    def test_property_12_structural_reward_with_thinking(self, output: str):
        """
        **Feature: medserl-adaptation, Property 12: Structural Reward Calculation**
        
        *For any* model output containing a <thinking> section, 
        the reward calculation SHALL include +0.1 structural reward.
        
        **Validates: Requirements 5.1**
        """
        # Use a clean ground truth to isolate structural reward effect
        ground_truth = {"has_error": False}
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.structural_reward == pytest.approx(STRUCTURAL_REWARD), \
            f"Output with <thinking> should have structural reward of {STRUCTURAL_REWARD}"

    @settings(max_examples=100)
    @given(output=output_without_thinking())
    def test_property_12_no_structural_reward_without_thinking(self, output: str):
        """
        **Feature: medserl-adaptation, Property 12: Structural Reward Calculation**
        
        *For any* model output NOT containing a <thinking> section,
        the structural reward SHALL be 0.0.
        
        **Validates: Requirements 5.1**
        """
        ground_truth = {"has_error": False}
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.structural_reward == pytest.approx(0.0), \
            "Output without <thinking> should have structural reward of 0.0"

    @settings(max_examples=100)
    @given(error_type=error_type_strategy, thinking=thinking_content_strategy)
    def test_property_13_correct_error_detection_reward(self, error_type: str, thinking: str):
        """
        **Feature: medserl-adaptation, Property 13: Correct Error Detection Reward**
        
        *For any* prediction on an error note that correctly identifies 
        the specific error type, the reward calculation SHALL include +1.0 outcome reward.
        
        **Validates: Requirements 5.2**
        """
        output = f"<thinking>{thinking}</thinking><verdict>Error: {error_type}</verdict>"
        ground_truth = {"has_error": True, "error_type": error_type}
        
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.outcome_reward == pytest.approx(CORRECT_CLASSIFICATION_REWARD), \
            f"Correct error detection should have outcome reward of {CORRECT_CLASSIFICATION_REWARD}"
        assert metadata.correct_classification is True, \
            "Correct error detection should be marked as correct classification"

    @settings(max_examples=100)
    @given(
        ground_truth=error_ground_truth(),
        thinking=thinking_content_strategy
    )
    def test_property_14_false_negative_penalty(self, ground_truth: dict, thinking: str):
        """
        **Feature: medserl-adaptation, Property 14: False Negative Penalty**
        
        *For any* prediction on an error note that states "No Error",
        the reward calculation SHALL include -1.0 penalty.
        
        **Validates: Requirements 5.3**
        """
        # Output says "No Clinical Error" but ground truth has an error
        output = f"<thinking>{thinking}</thinking><verdict>No Clinical Error</verdict>"
        
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.outcome_reward == pytest.approx(FALSE_NEGATIVE_PENALTY), \
            f"False negative should have outcome reward of {FALSE_NEGATIVE_PENALTY}"
        assert metadata.false_negative is True, \
            "False negative should be marked as false_negative"
        assert metadata.correct_classification is False, \
            "False negative should not be marked as correct classification"

    @settings(max_examples=100)
    @given(thinking=thinking_content_strategy)
    def test_property_15_correct_clean_classification_reward(self, thinking: str):
        """
        **Feature: medserl-adaptation, Property 15: Correct Clean Classification Reward**
        
        *For any* prediction on a clean note that states "No Clinical Error",
        the reward calculation SHALL include +1.0 outcome reward.
        
        **Validates: Requirements 5.4**
        """
        output = f"<thinking>{thinking}</thinking><verdict>No Clinical Error</verdict>"
        ground_truth = {"has_error": False}
        
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.outcome_reward == pytest.approx(CORRECT_CLASSIFICATION_REWARD), \
            f"Correct clean classification should have outcome reward of {CORRECT_CLASSIFICATION_REWARD}"
        assert metadata.correct_classification is True, \
            "Correct clean classification should be marked as correct"

    @settings(max_examples=100)
    @given(error_type=error_type_strategy, thinking=thinking_content_strategy)
    def test_property_16_false_positive_penalty(self, error_type: str, thinking: str):
        """
        **Feature: medserl-adaptation, Property 16: False Positive Penalty**
        
        *For any* prediction on a clean note that incorrectly identifies an error,
        the reward calculation SHALL include -1.5 penalty.
        
        **Validates: Requirements 5.5**
        """
        # Output says there's an error but ground truth is clean
        output = f"<thinking>{thinking}</thinking><verdict>Error: {error_type}</verdict>"
        ground_truth = {"has_error": False}
        
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        assert metadata.outcome_reward == pytest.approx(FALSE_POSITIVE_PENALTY), \
            f"False positive should have outcome reward of {FALSE_POSITIVE_PENALTY}"
        assert metadata.false_positive is True, \
            "False positive should be marked as false_positive"
        assert metadata.correct_classification is False, \
            "False positive should not be marked as correct classification"

    @settings(max_examples=100)
    @given(
        has_thinking=st.booleans(),
        has_error=st.booleans(),
        error_type=error_type_strategy,
        thinking=thinking_content_strategy,
        predict_error=st.booleans(),
        predicted_type=error_type_strategy
    )
    def test_property_17_reward_additivity(
        self,
        has_thinking: bool,
        has_error: bool,
        error_type: str,
        thinking: str,
        predict_error: bool,
        predicted_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 17: Reward Additivity**
        
        *For any* reward calculation, the total reward SHALL equal 
        the sum of structural_reward + outcome_reward components.
        
        **Validates: Requirements 5.6**
        """
        # Build output based on parameters
        if has_thinking:
            if predict_error:
                output = f"<thinking>{thinking}</thinking><verdict>Error: {predicted_type}</verdict>"
            else:
                output = f"<thinking>{thinking}</thinking><verdict>No Clinical Error</verdict>"
        else:
            if predict_error:
                output = f"<verdict>Error: {predicted_type}</verdict>"
            else:
                output = f"<verdict>No Clinical Error</verdict>"
        
        # Build ground truth
        if has_error:
            ground_truth = {"has_error": True, "error_type": error_type}
        else:
            ground_truth = {"has_error": False}
        
        metadata = calculate_reward_with_metadata(output, ground_truth)
        
        # The key property: total = structural + outcome
        expected_total = metadata.structural_reward + metadata.outcome_reward
        
        assert metadata.total_reward == pytest.approx(expected_total), \
            f"Total reward ({metadata.total_reward}) should equal " \
            f"structural ({metadata.structural_reward}) + outcome ({metadata.outcome_reward})"


class TestHasThinkingSection:
    """Tests for has_thinking_section helper function."""

    def test_with_thinking_section(self):
        """Output with <thinking> section returns True."""
        output = "<thinking>Analysis here</thinking><verdict>No Error</verdict>"
        assert has_thinking_section(output) is True

    def test_without_thinking_section(self):
        """Output without <thinking> section returns False."""
        output = "<verdict>No Clinical Error</verdict>"
        assert has_thinking_section(output) is False

    def test_empty_output(self):
        """Empty output returns False."""
        assert has_thinking_section("") is False
        assert has_thinking_section(None) is False

    def test_case_insensitive(self):
        """Thinking section detection is case-insensitive."""
        output = "<THINKING>Analysis</THINKING><verdict>No Error</verdict>"
        assert has_thinking_section(output) is True

    def test_multiline_thinking(self):
        """Multiline thinking section is detected."""
        output = """<thinking>
        Line 1
        Line 2
        </thinking>
        <verdict>No Clinical Error</verdict>"""
        assert has_thinking_section(output) is True


class TestParseVerdict:
    """Tests for parse_verdict helper function."""

    def test_extract_verdict(self):
        """Extracts verdict content correctly."""
        output = "<thinking>Analysis</thinking><verdict>Error: Diagnosis</verdict>"
        assert parse_verdict(output) == "Error: Diagnosis"

    def test_no_verdict(self):
        """Returns None when no verdict section."""
        output = "<thinking>Analysis only</thinking>"
        assert parse_verdict(output) is None

    def test_empty_output(self):
        """Returns None for empty output."""
        assert parse_verdict("") is None
        assert parse_verdict(None) is None

    def test_case_insensitive(self):
        """Verdict extraction is case-insensitive."""
        output = "<VERDICT>No Clinical Error</VERDICT>"
        assert parse_verdict(output) == "No Clinical Error"

    def test_strips_whitespace(self):
        """Strips whitespace from verdict content."""
        output = "<verdict>  Error: Treatment  </verdict>"
        assert parse_verdict(output) == "Error: Treatment"


class TestCalculateReward:
    """Tests for calculate_reward function."""

    def test_structural_reward_with_thinking(self):
        """Awards structural reward when <thinking> section present."""
        output = "<thinking>Analysis</thinking><verdict>No Clinical Error</verdict>"
        ground_truth = {"has_error": False}
        reward = calculate_reward(output, ground_truth)
        # Should get structural (0.1) + correct clean (1.0) = 1.1
        assert reward == pytest.approx(1.1)

    def test_no_structural_reward_without_thinking(self):
        """No structural reward when <thinking> section missing."""
        output = "<verdict>No Clinical Error</verdict>"
        ground_truth = {"has_error": False}
        reward = calculate_reward(output, ground_truth)
        # Should get only correct clean (1.0)
        assert reward == pytest.approx(1.0)

    def test_correct_error_detection(self):
        """Awards +1.0 for correctly identifying error type."""
        output = (
            "<thinking>Found issue</thinking>"
            "<verdict>Error: Pharmacotherapy</verdict>"
        )
        ground_truth = {"has_error": True, "error_type": "Pharmacotherapy"}
        reward = calculate_reward(output, ground_truth)
        # structural (0.1) + correct error (1.0) = 1.1
        assert reward == pytest.approx(1.1)

    def test_false_negative_penalty(self):
        """Penalizes -1.0 for missing an error."""
        output = (
            "<thinking>Looks fine</thinking>"
            "<verdict>No Clinical Error</verdict>"
        )
        ground_truth = {"has_error": True, "error_type": "Diagnosis"}
        reward = calculate_reward(output, ground_truth)
        # structural (0.1) + false negative (-1.0) = -0.9
        assert reward == pytest.approx(-0.9)

    def test_correct_clean_classification(self):
        """Awards +1.0 for correctly identifying clean note."""
        output = (
            "<thinking>No issues found</thinking>"
            "<verdict>No Clinical Error</verdict>"
        )
        ground_truth = {"has_error": False}
        reward = calculate_reward(output, ground_truth)
        # structural (0.1) + correct clean (1.0) = 1.1
        assert reward == pytest.approx(1.1)

    def test_false_positive_penalty(self):
        """Penalizes -1.5 for incorrectly flagging clean note."""
        output = (
            "<thinking>Found something</thinking>"
            "<verdict>Error: Treatment</verdict>"
        )
        ground_truth = {"has_error": False}
        reward = calculate_reward(output, ground_truth)
        # structural (0.1) + false positive (-1.5) = -1.4
        assert reward == pytest.approx(-1.4)

    def test_wrong_error_type(self):
        """No outcome reward for wrong error type detection."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>Error: Diagnosis</verdict>"
        )
        ground_truth = {"has_error": True, "error_type": "Pharmacotherapy"}
        reward = calculate_reward(output, ground_truth)
        # structural (0.1) + wrong type (0.0) = 0.1
        assert reward == pytest.approx(0.1)


class TestCalculateRewardWithMetadata:
    """Tests for calculate_reward_with_metadata function."""

    def test_returns_metadata_object(self):
        """Returns RewardMetadata dataclass."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>No Clinical Error</verdict>"
        )
        ground_truth = {"has_error": False}
        result = calculate_reward_with_metadata(output, ground_truth)
        assert isinstance(result, RewardMetadata)

    def test_metadata_correct_classification(self):
        """Metadata correctly indicates correct classification."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>Error: Diagnosis</verdict>"
        )
        ground_truth = {"has_error": True, "error_type": "Diagnosis"}
        result = calculate_reward_with_metadata(output, ground_truth)
        assert result.correct_classification is True
        assert result.false_positive is False
        assert result.false_negative is False

    def test_metadata_false_positive(self):
        """Metadata correctly indicates false positive."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>Error: Treatment</verdict>"
        )
        ground_truth = {"has_error": False}
        result = calculate_reward_with_metadata(output, ground_truth)
        assert result.correct_classification is False
        assert result.false_positive is True
        assert result.false_negative is False

    def test_metadata_false_negative(self):
        """Metadata correctly indicates false negative."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>No Clinical Error</verdict>"
        )
        ground_truth = {"has_error": True, "error_type": "Diagnosis"}
        result = calculate_reward_with_metadata(output, ground_truth)
        assert result.correct_classification is False
        assert result.false_positive is False
        assert result.false_negative is True

    def test_metadata_reward_additivity(self):
        """Total reward equals sum of components."""
        output = (
            "<thinking>Analysis</thinking>"
            "<verdict>No Clinical Error</verdict>"
        )
        ground_truth = {"has_error": False}
        result = calculate_reward_with_metadata(output, ground_truth)
        expected_total = result.structural_reward + result.outcome_reward
        assert result.total_reward == pytest.approx(expected_total)
