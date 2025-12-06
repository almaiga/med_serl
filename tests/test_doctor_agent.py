"""
Property-based tests for MedSeRL DoctorAgent.

**Feature: medserl-adaptation, Property 10: Doctor Output Format Validity**
**Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
"""

import re
from typing import Optional

from hypothesis import given, strategies as st, settings

from src.agents.doctor_agent import (
    MockDoctorAgent,
    ERROR_TYPES,
    DoctorPrediction
)


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Valid MEDEC error types
VALID_ERROR_TYPES = ERROR_TYPES

# Strategy for generating clinical note text
clinical_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' .,;:-'
    ),
    min_size=10,
    max_size=500
).filter(lambda s: s.strip())

# Strategy for generating error types
error_type_strategy = st.sampled_from(VALID_ERROR_TYPES)

# Strategy for whether the note has an error
has_error_strategy = st.booleans()


def has_thinking_section(output: str) -> bool:
    """Check if output contains a <thinking> section."""
    return bool(re.search(r'<thinking>.*?</thinking>', output, re.DOTALL | re.IGNORECASE))


def has_verdict_section(output: str) -> bool:
    """Check if output contains a <verdict> section."""
    return bool(re.search(r'<verdict>.*?</verdict>', output, re.DOTALL | re.IGNORECASE))


def get_section_positions(output: str) -> tuple:
    """
    Get the start positions of <thinking> and <verdict> sections.
    
    Returns:
        Tuple of (thinking_start, verdict_start) or (-1, -1) if not found
    """
    thinking_match = re.search(r'<thinking>', output, re.IGNORECASE)
    verdict_match = re.search(r'<verdict>', output, re.IGNORECASE)
    
    thinking_start = thinking_match.start() if thinking_match else -1
    verdict_start = verdict_match.start() if verdict_match else -1
    
    return (thinking_start, verdict_start)


# =============================================================================
# Property 10: Doctor Output Format Validity
# **Feature: medserl-adaptation, Property 10: Doctor Output Format Validity**
# **Validates: Requirements 4.1**
#
# For any clinical note input to the Doctor Agent, the output SHALL contain
# both <thinking> and <verdict> sections.
# =============================================================================

class TestDoctorOutputFormatValidity:
    """Property tests for Doctor output format validity (Property 10)."""

    @given(
        note=clinical_text_strategy,
        has_error=has_error_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_output_contains_thinking_section(
        self,
        note: str,
        has_error: bool,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 10: Doctor Output Format Validity**
        **Validates: Requirements 4.1**

        For any clinical note input, the Doctor Agent output SHALL contain
        a <thinking> section.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(
            has_error=has_error,
            error_type=error_type if has_error else None
        )
        
        output = agent.analyze_note(note)
        
        assert has_thinking_section(output), (
            f"Output must contain <thinking> section.\n"
            f"Got output: {output[:200]}..."
        )

    @given(
        note=clinical_text_strategy,
        has_error=has_error_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_output_contains_verdict_section(
        self,
        note: str,
        has_error: bool,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 10: Doctor Output Format Validity**
        **Validates: Requirements 4.1**

        For any clinical note input, the Doctor Agent output SHALL contain
        a <verdict> section.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(
            has_error=has_error,
            error_type=error_type if has_error else None
        )
        
        output = agent.analyze_note(note)
        
        assert has_verdict_section(output), (
            f"Output must contain <verdict> section.\n"
            f"Got output: {output[:200]}..."
        )

    @given(
        note=clinical_text_strategy,
        has_error=has_error_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_output_contains_both_sections(
        self,
        note: str,
        has_error: bool,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 10: Doctor Output Format Validity**
        **Validates: Requirements 4.1**

        For any clinical note input, the Doctor Agent output SHALL contain
        BOTH <thinking> and <verdict> sections.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(
            has_error=has_error,
            error_type=error_type if has_error else None
        )
        
        output = agent.analyze_note(note)
        
        has_thinking = has_thinking_section(output)
        has_verdict = has_verdict_section(output)
        
        assert has_thinking and has_verdict, (
            f"Output must contain both <thinking> and <verdict> sections.\n"
            f"Has <thinking>: {has_thinking}\n"
            f"Has <verdict>: {has_verdict}\n"
            f"Got output: {output[:200]}..."
        )


# =============================================================================
# Property 11: Thinking Section Ordering
# **Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
# **Validates: Requirements 4.5**
#
# For any Doctor Agent output, the <thinking> section SHALL appear before
# the <verdict> section.
# =============================================================================

class TestThinkingSectionOrdering:
    """Property tests for thinking section ordering (Property 11)."""

    @given(
        note=clinical_text_strategy,
        has_error=has_error_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_thinking_precedes_verdict(
        self,
        note: str,
        has_error: bool,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
        **Validates: Requirements 4.5**

        For any Doctor Agent output, the <thinking> section SHALL appear
        before the <verdict> section.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(
            has_error=has_error,
            error_type=error_type if has_error else None
        )
        
        output = agent.analyze_note(note)
        
        thinking_start, verdict_start = get_section_positions(output)
        
        # Both sections must exist
        assert thinking_start >= 0, (
            f"Output must contain <thinking> section.\n"
            f"Got output: {output[:200]}..."
        )
        assert verdict_start >= 0, (
            f"Output must contain <verdict> section.\n"
            f"Got output: {output[:200]}..."
        )
        
        # Thinking must come before verdict
        assert thinking_start < verdict_start, (
            f"<thinking> section (at position {thinking_start}) must appear "
            f"before <verdict> section (at position {verdict_start}).\n"
            f"Got output: {output[:200]}..."
        )

    @given(
        notes=st.lists(clinical_text_strategy, min_size=1, max_size=5),
        has_errors=st.lists(has_error_strategy, min_size=1, max_size=5),
        error_types=st.lists(error_type_strategy, min_size=1, max_size=5)
    )
    @settings(max_examples=100)
    def test_batch_outputs_maintain_ordering(
        self,
        notes: list,
        has_errors: list,
        error_types: list
    ):
        """
        **Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
        **Validates: Requirements 4.5**

        For any batch of Doctor Agent outputs, ALL outputs SHALL have
        <thinking> section appearing before <verdict> section.
        """
        agent = MockDoctorAgent()
        
        # Process each note individually (since mock doesn't support batch config per-item)
        for i, (note, has_error, error_type) in enumerate(
            zip(notes, has_errors, error_types)
        ):
            agent.set_mock_response(
                has_error=has_error,
                error_type=error_type if has_error else None
            )
            
            output = agent.analyze_note(note)
            thinking_start, verdict_start = get_section_positions(output)
            
            assert thinking_start >= 0 and verdict_start >= 0, (
                f"Output {i} must contain both sections"
            )
            assert thinking_start < verdict_start, (
                f"Output {i}: <thinking> (at {thinking_start}) must appear "
                f"before <verdict> (at {verdict_start})"
            )

    @given(
        note=clinical_text_strategy
    )
    @settings(max_examples=100)
    def test_ordering_for_clean_notes(
        self,
        note: str
    ):
        """
        **Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
        **Validates: Requirements 4.5**

        For any clean note (no error), the <thinking> section SHALL still
        appear before the <verdict> section.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(has_error=False, error_type=None)
        
        output = agent.analyze_note(note)
        
        thinking_start, verdict_start = get_section_positions(output)
        
        assert thinking_start >= 0 and verdict_start >= 0, (
            "Output must contain both <thinking> and <verdict> sections"
        )
        assert thinking_start < verdict_start, (
            f"<thinking> (at {thinking_start}) must appear before "
            f"<verdict> (at {verdict_start})"
        )

    @given(
        note=clinical_text_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_ordering_for_error_notes(
        self,
        note: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 11: Thinking Section Ordering**
        **Validates: Requirements 4.5**

        For any error note, the <thinking> section SHALL still appear
        before the <verdict> section.
        """
        agent = MockDoctorAgent()
        agent.set_mock_response(has_error=True, error_type=error_type)
        
        output = agent.analyze_note(note)
        
        thinking_start, verdict_start = get_section_positions(output)
        
        assert thinking_start >= 0 and verdict_start >= 0, (
            "Output must contain both <thinking> and <verdict> sections"
        )
        assert thinking_start < verdict_start, (
            f"<thinking> (at {thinking_start}) must appear before "
            f"<verdict> (at {verdict_start})"
        )
