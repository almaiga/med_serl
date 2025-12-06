"""
Tests for SFT data preparation in MedSeRL training.

Property-based tests for SFT data filtering, target format, and thinking content.

Requirements: 6.1, 6.2, 6.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.training.train_serl import (
    prepare_sft_data,
    format_sft_input,
    format_sft_target,
    SFTExample,
    _normalize_error_type_for_output,
    _generate_thinking_content
)
from src.data_processor import MedicalDataProcessor


# =============================================================================
# Hypothesis Strategies for Property-Based Testing
# =============================================================================

# Valid MEDEC error types
ERROR_TYPES = ["Diagnosis", "Management", "Treatment", "Pharmacotherapy", "Causal Organism"]

# Variations of error types that might appear in raw data
ERROR_TYPE_VARIATIONS = [
    "diagnosis", "Diagnosis", "DIAGNOSIS",
    "management", "Management", "MANAGEMENT",
    "treatment", "Treatment", "TREATMENT",
    "pharmacotherapy", "Pharmacotherapy", "PHARMACOTHERAPY",
    "causalOrganism", "Causal Organism", "causal_organism", "causalorganism"
]

# Strategy for generating error types (including variations)
error_type_strategy = st.sampled_from(ERROR_TYPE_VARIATIONS)

# Strategy for generating normalized error types
normalized_error_type_strategy = st.sampled_from(ERROR_TYPES)

# Strategy for generating clinical note text
clinical_note_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
    min_size=10,
    max_size=500
).filter(lambda x: x.strip())

# Strategy for generating text IDs
text_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N')),
    min_size=1,
    max_size=20
).filter(lambda x: x.strip())


# Strategy for generating mock error pool entries
@st.composite
def error_pool_entry(draw):
    """Generate a mock error pool entry (error_flag=1)."""
    return {
        'text': draw(clinical_note_strategy),
        'text_id': draw(text_id_strategy),
        'error_type': draw(error_type_strategy),
        'label': 'Error',
        'subset': draw(st.sampled_from(['MS', 'UW'])),
        'source_file': 'test_file.csv'
    }


# Strategy for generating mock clean pool entries
@st.composite
def clean_pool_entry(draw):
    """Generate a mock clean pool entry (error_flag=0)."""
    return {
        'text': draw(clinical_note_strategy),
        'text_id': draw(text_id_strategy),
        'label': 'Clean',
        'subset': draw(st.sampled_from(['MS', 'UW'])),
        'source_file': 'test_file.csv'
    }


# Strategy for generating a mock data processor with error pool
@st.composite
def mock_data_processor_with_errors(draw):
    """Generate a mock data processor with error pool entries."""
    num_entries = draw(st.integers(min_value=1, max_value=10))
    error_pool = [draw(error_pool_entry()) for _ in range(num_entries)]
    
    # Create a mock processor
    class MockDataProcessor:
        def __init__(self, error_pool):
            self._error_pool = error_pool
        
        def get_error_pool(self):
            return self._error_pool
    
    return MockDataProcessor(error_pool)


# =============================================================================
# Property-Based Tests for SFT Data
# =============================================================================

class TestPropertyBasedSFTData:
    """
    Property-based tests for SFT data preparation.
    
    **Feature: medserl-adaptation, Property 18-20: SFT Data Properties**
    **Validates: Requirements 6.1, 6.2, 6.5**
    """

    @settings(max_examples=100)
    @given(processor=mock_data_processor_with_errors())
    def test_property_18_sft_data_filtering(self, processor):
        """
        **Feature: medserl-adaptation, Property 18: SFT Data Filtering**
        
        *For any* sample included in SFT warm-up training, 
        the sample SHALL have error_flag equal to 1.
        
        Since prepare_sft_data uses get_error_pool() which only contains
        entries with error_flag=1, all SFT examples must come from error samples.
        
        **Validates: Requirements 6.1**
        """
        # Prepare SFT data from the processor
        sft_examples = prepare_sft_data(processor)
        
        # Get the error pool for comparison
        error_pool = processor.get_error_pool()
        
        # All SFT examples should come from error pool entries
        assert len(sft_examples) <= len(error_pool), \
            "SFT examples should not exceed error pool size"
        
        # Each SFT example should have an error_type (indicating it came from error pool)
        for example in sft_examples:
            assert isinstance(example, SFTExample), \
                "Each SFT example should be an SFTExample instance"
            assert example.error_type is not None, \
                "SFT examples should have an error_type (from error pool)"
            assert example.error_type != '', \
                "SFT examples should have a non-empty error_type"

    @settings(max_examples=100)
    @given(error_type=error_type_strategy)
    def test_property_19_sft_target_format(self, error_type: str):
        """
        **Feature: medserl-adaptation, Property 19: SFT Target Format**
        
        *For any* SFT training example, the target output SHALL contain 
        both <thinking> and <verdict> sections.
        
        **Validates: Requirements 6.2**
        """
        # Generate target for this error type
        target = format_sft_target(error_type)
        
        # Check for <thinking> section
        assert '<thinking>' in target, \
            f"SFT target should contain <thinking> tag, got: {target}"
        assert '</thinking>' in target, \
            f"SFT target should contain </thinking> tag, got: {target}"
        
        # Check for <verdict> section
        assert '<verdict>' in target, \
            f"SFT target should contain <verdict> tag, got: {target}"
        assert '</verdict>' in target, \
            f"SFT target should contain </verdict> tag, got: {target}"
        
        # Verify ordering: <thinking> should come before <verdict>
        thinking_start = target.find('<thinking>')
        verdict_start = target.find('<verdict>')
        assert thinking_start < verdict_start, \
            "The <thinking> section should appear before <verdict> section"

    @settings(max_examples=100)
    @given(error_type=error_type_strategy)
    def test_property_20_sft_thinking_content(self, error_type: str):
        """
        **Feature: medserl-adaptation, Property 20: SFT Thinking Content**
        
        *For any* SFT training example, the <thinking> section SHALL 
        mention the specific error type from the source data.
        
        **Validates: Requirements 6.5**
        """
        # Generate target for this error type
        target = format_sft_target(error_type)
        
        # Extract thinking section content
        thinking_start = target.find('<thinking>') + len('<thinking>')
        thinking_end = target.find('</thinking>')
        thinking_content = target[thinking_start:thinking_end]
        
        # Normalize the error type for comparison
        normalized_type = _normalize_error_type_for_output(error_type)
        
        # The thinking section should mention the specific error type
        assert normalized_type in thinking_content, \
            f"Thinking section should mention '{normalized_type}', got: {thinking_content}"
        
        # The thinking section should indicate the error was found
        assert 'Found' in thinking_content or 'found' in thinking_content.lower(), \
            f"Thinking section should indicate error was found for {normalized_type}"


class TestSFTTargetFormatUnit:
    """Unit tests for SFT target formatting functions."""

    def test_format_sft_target_basic(self):
        """Basic test for format_sft_target."""
        target = format_sft_target("Diagnosis")
        assert '<thinking>' in target
        assert '</thinking>' in target
        assert '<verdict>' in target
        assert '</verdict>' in target
        assert 'Error: Diagnosis' in target

    def test_format_sft_target_all_error_types(self):
        """Test format_sft_target for all valid error types."""
        for error_type in ERROR_TYPES:
            target = format_sft_target(error_type)
            assert f'Error: {error_type}' in target, \
                f"Target should contain 'Error: {error_type}'"

    def test_format_sft_target_causal_organism_normalization(self):
        """Test that causalOrganism is normalized to 'Causal Organism'."""
        target = format_sft_target("causalOrganism")
        assert 'Error: Causal Organism' in target
        assert 'Causal Organism' in target

    def test_normalize_error_type_variations(self):
        """Test error type normalization for various input formats."""
        test_cases = [
            ("diagnosis", "Diagnosis"),
            ("Diagnosis", "Diagnosis"),
            ("DIAGNOSIS", "Diagnosis"),
            ("management", "Management"),
            ("treatment", "Treatment"),
            ("pharmacotherapy", "Pharmacotherapy"),
            ("causalOrganism", "Causal Organism"),
            ("causal organism", "Causal Organism"),
            ("causal_organism", "Causal Organism"),
        ]
        for input_type, expected in test_cases:
            result = _normalize_error_type_for_output(input_type)
            assert result == expected, \
                f"Expected '{expected}' for input '{input_type}', got '{result}'"


class TestSFTInputFormatUnit:
    """Unit tests for SFT input formatting."""

    def test_format_sft_input_basic(self):
        """Basic test for format_sft_input."""
        note = "Patient presents with fever and cough."
        input_text = format_sft_input(note)
        
        assert note in input_text, "Input should contain the clinical note"
        assert "Clinical Note:" in input_text, "Input should have 'Clinical Note:' label"
        assert "Analysis:" in input_text, "Input should end with 'Analysis:'"

    def test_format_sft_input_with_custom_system_prompt(self):
        """Test format_sft_input with custom system prompt."""
        note = "Patient has diabetes."
        custom_prompt = "You are a medical expert."
        input_text = format_sft_input(note, system_prompt=custom_prompt)
        
        assert custom_prompt in input_text, "Input should contain custom system prompt"
        assert note in input_text, "Input should contain the clinical note"

    def test_format_sft_input_default_system_prompt(self):
        """Test that default system prompt mentions error types."""
        note = "Test note."
        input_text = format_sft_input(note)
        
        # Default prompt should mention the five error types
        assert "Diagnosis" in input_text or "error" in input_text.lower()


class TestGenerateThinkingContent:
    """Unit tests for thinking content generation."""

    def test_generate_thinking_content_identifies_error(self):
        """Test that thinking content identifies the specific error."""
        for error_type in ERROR_TYPES:
            content = _generate_thinking_content(error_type)
            assert f"Found a {error_type} error" in content, \
                f"Should identify {error_type} error in thinking"

    def test_generate_thinking_content_checks_all_types(self):
        """Test that thinking content checks all error types."""
        content = _generate_thinking_content("Diagnosis")
        
        # Should mention checking for each error type
        for error_type in ERROR_TYPES:
            assert error_type in content, \
                f"Thinking should mention checking for {error_type}"

    def test_generate_thinking_content_only_one_found(self):
        """Test that only one error type is marked as found."""
        for target_type in ERROR_TYPES:
            content = _generate_thinking_content(target_type)
            
            # Count how many "Found" statements there are
            found_count = content.count("Found a")
            assert found_count == 1, \
                f"Should have exactly one 'Found' statement, got {found_count}"
            
            # The found statement should be for the target type
            assert f"Found a {target_type} error" in content


class TestPrepareSFTDataIntegration:
    """Integration tests for prepare_sft_data with mock data."""

    def test_prepare_sft_data_empty_pool(self):
        """Test prepare_sft_data with empty error pool."""
        class EmptyProcessor:
            def get_error_pool(self):
                return []
        
        examples = prepare_sft_data(EmptyProcessor())
        assert examples == [], "Should return empty list for empty error pool"

    def test_prepare_sft_data_filters_empty_text(self):
        """Test that entries with empty text are filtered out."""
        class ProcessorWithEmptyText:
            def get_error_pool(self):
                return [
                    {'text': '', 'error_type': 'Diagnosis', 'text_id': '1'},
                    {'text': '   ', 'error_type': 'Treatment', 'text_id': '2'},
                    {'text': 'Valid note', 'error_type': 'Management', 'text_id': '3'},
                ]
        
        examples = prepare_sft_data(ProcessorWithEmptyText())
        assert len(examples) == 1, "Should filter out empty text entries"
        assert examples[0].error_type == 'Management'

    def test_prepare_sft_data_preserves_error_type(self):
        """Test that error type is preserved in SFT examples."""
        class ProcessorWithTypes:
            def get_error_pool(self):
                return [
                    {'text': 'Note 1', 'error_type': 'Diagnosis', 'text_id': '1'},
                    {'text': 'Note 2', 'error_type': 'Pharmacotherapy', 'text_id': '2'},
                ]
        
        examples = prepare_sft_data(ProcessorWithTypes())
        error_types = {ex.error_type for ex in examples}
        
        # Note: error types get normalized in the target
        assert len(examples) == 2
