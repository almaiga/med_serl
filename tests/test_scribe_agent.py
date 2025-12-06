"""
Property-based tests for MedSeRL ScribeAgent.

**Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
"""

from typing import Dict, List, Optional

from hypothesis import given, strategies as st, settings

from src.agents.scribe_agent import MockScribeAgent


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Valid MEDEC error types
VALID_ERROR_TYPES = [
    'Diagnosis', 'Management', 'Treatment',
    'Pharmacotherapy', 'Causal Organism'
]

# Quadrant modes and their expected has_error values
MODE_HAS_ERROR_MAP = {
    'augment_error': True,        # Augmented Ground Truth - preserves error
    'augment_safe': False,        # Augmented Safe - clean note paraphrased
    'make_decoy': False,          # Synthetic Decoy - cosmetic noise only
    'inject_new_error': True,     # Synthetic Injection - new error injected
}

# Strategy for generating clinical note text
clinical_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' .,;:-'
    ),
    min_size=10,
    max_size=200
).filter(lambda s: s.strip())

# Strategy for generating error types
error_type_strategy = st.sampled_from(VALID_ERROR_TYPES)

# Strategy for generating transformation modes
mode_strategy = st.sampled_from(list(MODE_HAS_ERROR_MAP.keys()))


def create_prompt_dict(
    text: str,
    mode: str,
    has_error: bool,
    error_type: Optional[str] = None,
    source: Optional[str] = None
) -> Dict:
    """Create a prompt dictionary as expected by ScribeAgent.transform_batch."""
    meta = {
        "has_error": has_error,
        "source": source or mode
    }
    if error_type:
        meta["error_type"] = error_type

    return {
        "scribe_prompt": f"Task: Transform this note.\nInput Note: {text}",
        "meta": meta,
        "original_text": text,
        "mode": mode
    }


# =============================================================================
# Property 9: Ground Truth Metadata Attachment
# **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
# **Validates: Requirements 3.6**
#
# For any transformation completed by the Scribe Agent, the output SHALL
# include ground truth metadata with has_error field correctly reflecting
# the transformation mode.
# =============================================================================

class TestGroundTruthMetadataAttachment:
    """Property tests for ground truth metadata attachment (Property 9)."""

    @given(
        text=clinical_text_strategy,
        mode=mode_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_transform_attaches_ground_truth_metadata(
        self,
        text: str,
        mode: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any transformation completed by the Scribe Agent, the output
        SHALL include ground truth metadata.
        """
        expected_has_error = MODE_HAS_ERROR_MAP[mode]

        # Create prompt with appropriate metadata
        prompt = create_prompt_dict(
            text=text,
            mode=mode,
            has_error=expected_has_error,
            error_type=error_type if expected_has_error else None
        )

        # Use MockScribeAgent for testing without GPU
        agent = MockScribeAgent()
        results = agent.transform_batch([prompt])

        assert len(results) == 1
        result = results[0]

        # Verify ground_truth field exists
        assert 'ground_truth' in result, (
            "Result must contain 'ground_truth' field"
        )

        # Verify ground_truth has required structure
        ground_truth = result['ground_truth']
        assert isinstance(ground_truth, dict), (
            "ground_truth must be a dictionary"
        )
        assert 'has_error' in ground_truth, (
            "ground_truth must contain 'has_error' field"
        )

    @given(
        text=clinical_text_strategy,
        mode=mode_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_has_error_reflects_transformation_mode(
        self,
        text: str,
        mode: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any transformation, the has_error field SHALL correctly reflect
        whether the transformation mode produces an error-containing note.
        """
        expected_has_error = MODE_HAS_ERROR_MAP[mode]

        prompt = create_prompt_dict(
            text=text,
            mode=mode,
            has_error=expected_has_error,
            error_type=error_type if expected_has_error else None
        )

        agent = MockScribeAgent()
        results = agent.transform_batch([prompt])

        assert len(results) == 1
        ground_truth = results[0]['ground_truth']

        # Verify has_error matches expected value for the mode
        assert ground_truth['has_error'] == expected_has_error, (
            f"For mode '{mode}', has_error should be {expected_has_error}, "
            f"got {ground_truth['has_error']}"
        )

    @given(
        text=clinical_text_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_error_modes_have_error_type_in_metadata(
        self,
        text: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any transformation in error-producing modes (augment_error,
        inject_new_error), the ground truth SHALL include the error_type.
        """
        error_modes = ['augment_error', 'inject_new_error']

        for mode in error_modes:
            prompt = create_prompt_dict(
                text=text,
                mode=mode,
                has_error=True,
                error_type=error_type
            )

            agent = MockScribeAgent()
            results = agent.transform_batch([prompt])

            assert len(results) == 1
            ground_truth = results[0]['ground_truth']

            # Error modes should include error_type
            assert 'error_type' in ground_truth, (
                f"For error mode '{mode}', ground_truth must include error_type"
            )
            assert ground_truth['error_type'] == error_type, (
                f"error_type should be '{error_type}', "
                f"got '{ground_truth.get('error_type')}'"
            )

    @given(
        texts=st.lists(clinical_text_strategy, min_size=1, max_size=10),
        modes=st.lists(mode_strategy, min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_batch_transform_attaches_metadata_to_all(
        self,
        texts: List[str],
        modes: List[str]
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any batch of transformations, ALL outputs SHALL include
        ground truth metadata with correct has_error values.
        """
        # Create prompts for each text/mode pair
        prompts = []
        for text, mode in zip(texts, modes):
            expected_has_error = MODE_HAS_ERROR_MAP[mode]
            prompt = create_prompt_dict(
                text=text,
                mode=mode,
                has_error=expected_has_error,
                error_type='Diagnosis' if expected_has_error else None
            )
            prompts.append(prompt)

        agent = MockScribeAgent()
        results = agent.transform_batch(prompts)

        # Verify all results have metadata
        assert len(results) == len(prompts)

        for i, (result, prompt) in enumerate(zip(results, prompts)):
            expected_has_error = prompt['meta']['has_error']

            assert 'ground_truth' in result, (
                f"Result {i} must contain 'ground_truth' field"
            )
            assert 'has_error' in result['ground_truth'], (
                f"Result {i} ground_truth must contain 'has_error' field"
            )
            assert result['ground_truth']['has_error'] == expected_has_error, (
                f"Result {i}: has_error should be {expected_has_error}"
            )

    @given(
        text=clinical_text_strategy
    )
    @settings(max_examples=100)
    def test_clean_modes_do_not_have_error_type(
        self,
        text: str
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any transformation in clean modes (augment_safe, make_decoy),
        the ground truth has_error SHALL be False.
        """
        clean_modes = ['augment_safe', 'make_decoy']

        for mode in clean_modes:
            prompt = create_prompt_dict(
                text=text,
                mode=mode,
                has_error=False
            )

            agent = MockScribeAgent()
            results = agent.transform_batch([prompt])

            assert len(results) == 1
            ground_truth = results[0]['ground_truth']

            # Clean modes should have has_error=False
            assert ground_truth['has_error'] is False, (
                f"For clean mode '{mode}', has_error should be False"
            )

    @given(
        text=clinical_text_strategy,
        mode=mode_strategy
    )
    @settings(max_examples=100)
    def test_ground_truth_includes_source(
        self,
        text: str,
        mode: str
    ):
        """
        **Feature: medserl-adaptation, Property 9: Ground Truth Metadata Attachment**
        **Validates: Requirements 3.6**

        For any transformation, the ground truth SHALL include a source
        field indicating the transformation mode/origin.
        """
        expected_has_error = MODE_HAS_ERROR_MAP[mode]

        prompt = create_prompt_dict(
            text=text,
            mode=mode,
            has_error=expected_has_error,
            source=f"test_source_{mode}"
        )

        agent = MockScribeAgent()
        results = agent.transform_batch([prompt])

        assert len(results) == 1
        ground_truth = results[0]['ground_truth']

        # Verify source field exists
        assert 'source' in ground_truth, (
            "ground_truth must contain 'source' field"
        )
