"""
Tests for checkpoint module in MedSeRL training.

Property-based tests for checkpoint filename format.

Requirements: 8.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.training.checkpoint import (
    get_checkpoint_filename,
    save_checkpoint,
    list_checkpoints,
    CheckpointMetadata
)


# =============================================================================
# Hypothesis Strategies for Property-Based Testing
# =============================================================================

# Strategy for generating valid episode numbers (positive integers)
episode_strategy = st.integers(min_value=0, max_value=100000)


# =============================================================================
# Property-Based Tests for Checkpoint Filename
# =============================================================================

class TestPropertyBasedCheckpointFilename:
    """
    Property-based tests for checkpoint filename format.
    
    **Feature: medserl-adaptation, Property 21: Checkpoint Filename Format**
    **Validates: Requirements 8.6**
    """

    @settings(max_examples=100)
    @given(episode=episode_strategy)
    def test_property_21_checkpoint_filename_includes_episode(self, episode: int):
        """
        **Feature: medserl-adaptation, Property 21: Checkpoint Filename Format**
        
        *For any* saved checkpoint, the filename SHALL include the episode number.
        
        **Validates: Requirements 8.6**
        """
        # Generate checkpoint filename
        filename = get_checkpoint_filename(episode)
        
        # The filename must include the episode number
        assert str(episode) in filename, \
            f"Checkpoint filename '{filename}' should include episode number {episode}"
        
        # The filename should follow the expected pattern
        assert filename.startswith("checkpoint_episode_"), \
            f"Checkpoint filename should start with 'checkpoint_episode_', got: {filename}"
        
        # The episode number should be extractable from the filename
        extracted_episode = int(filename.replace("checkpoint_episode_", ""))
        assert extracted_episode == episode, \
            f"Extracted episode {extracted_episode} should match input episode {episode}"


class TestCheckpointFilenameUnit:
    """Unit tests for checkpoint filename generation."""

    def test_get_checkpoint_filename_basic(self):
        """Basic test for get_checkpoint_filename."""
        filename = get_checkpoint_filename(1)
        assert filename == "checkpoint_episode_1"

    def test_get_checkpoint_filename_zero(self):
        """Test checkpoint filename for episode 0."""
        filename = get_checkpoint_filename(0)
        assert filename == "checkpoint_episode_0"
        assert "0" in filename

    def test_get_checkpoint_filename_large_number(self):
        """Test checkpoint filename for large episode number."""
        filename = get_checkpoint_filename(99999)
        assert filename == "checkpoint_episode_99999"
        assert "99999" in filename

    def test_get_checkpoint_filename_format_consistency(self):
        """Test that filename format is consistent across episodes."""
        for episode in [1, 10, 100, 1000]:
            filename = get_checkpoint_filename(episode)
            assert filename == f"checkpoint_episode_{episode}"
