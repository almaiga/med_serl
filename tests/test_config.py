"""
Property-based tests for MedSeRL configuration validation.

**Feature: medserl-adaptation, Property 22: Path Validation**
**Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
"""

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, assume

from src.config import (
    MedSERLConfig,
    ModelPaths,
    DataPaths,
    TrainingHyperparameters,
    TrainingSchedule,
    InfrastructureConfig,
    RewardConfig,
    ConfigurationError,
    validate_path_exists,
)


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating non-existent paths
nonexistent_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-/'),
    min_size=5,
    max_size=50
).map(lambda s: f"/nonexistent_path_{s.replace('/', '_')}")


# Strategy for generating valid hyperparameters
valid_learning_rate = st.floats(min_value=1e-10, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_batch_size = st.integers(min_value=4, max_value=256).filter(lambda x: x % 4 == 0)
valid_kl_coef = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
valid_gamma = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_clip_range = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for generating invalid hyperparameters
invalid_learning_rate = st.floats(max_value=0.0, allow_nan=False, allow_infinity=False)
invalid_batch_size_negative = st.integers(max_value=0)
invalid_batch_size_not_divisible = st.integers(min_value=1, max_value=255).filter(lambda x: x % 4 != 0)
invalid_kl_coef = st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False)
invalid_gamma = st.one_of(
    st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, allow_nan=False, allow_infinity=False)
)

# Strategy for valid positive integers
valid_positive_int = st.integers(min_value=1, max_value=10000)
invalid_positive_int = st.integers(max_value=0)


# =============================================================================
# Property 22: Path Validation
# **Feature: medserl-adaptation, Property 22: Path Validation**
# **Validates: Requirements 10.4**
#
# For any configuration with required paths, the system SHALL raise an error
# if any path does not exist.
# =============================================================================

class TestPathValidation:
    """Property tests for path validation (Property 22)."""

    @given(nonexistent_path=nonexistent_path_strategy)
    @settings(max_examples=100)
    def test_nonexistent_base_model_path_raises_error(self, nonexistent_path: str):
        """
        **Feature: medserl-adaptation, Property 22: Path Validation**
        **Validates: Requirements 10.4**
        
        For any non-existent base model path, validation SHALL return an error.
        """
        # Ensure the path doesn't actually exist
        assume(not Path(nonexistent_path).exists())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=nonexistent_path),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                )
            )
            
            errors = config.validate()
            
            # Should have at least one error about the path
            assert len(errors) > 0
            assert any("does not exist" in error for error in errors)
            assert any(nonexistent_path in error for error in errors)

    @given(nonexistent_path=nonexistent_path_strategy)
    @settings(max_examples=100)
    def test_nonexistent_medec_data_path_raises_error(self, nonexistent_path: str):
        """
        **Feature: medserl-adaptation, Property 22: Path Validation**
        **Validates: Requirements 10.4**
        
        For any non-existent MEDEC data path, validation SHALL return an error.
        """
        assume(not Path(nonexistent_path).exists())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=nonexistent_path,
                    output_dir=tmpdir
                )
            )
            
            errors = config.validate()
            
            assert len(errors) > 0
            assert any("does not exist" in error for error in errors)
            assert any(nonexistent_path in error for error in errors)

    @given(nonexistent_path=nonexistent_path_strategy)
    @settings(max_examples=100)
    def test_validate_path_exists_raises_configuration_error(self, nonexistent_path: str):
        """
        **Feature: medserl-adaptation, Property 22: Path Validation**
        **Validates: Requirements 10.4**
        
        For any non-existent path, validate_path_exists SHALL raise ConfigurationError.
        """
        assume(not Path(nonexistent_path).exists())
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_path_exists(nonexistent_path, "Test path")
        
        assert nonexistent_path in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)

    @given(nonexistent_path=nonexistent_path_strategy)
    @settings(max_examples=100)
    def test_validate_or_raise_raises_on_invalid_paths(self, nonexistent_path: str):
        """
        **Feature: medserl-adaptation, Property 22: Path Validation**
        **Validates: Requirements 10.4**
        
        For any configuration with non-existent required paths,
        validate_or_raise SHALL raise ConfigurationError.
        """
        assume(not Path(nonexistent_path).exists())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=nonexistent_path),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                )
            )
            
            with pytest.raises(ConfigurationError):
                config.validate_or_raise()

    def test_valid_paths_pass_validation(self):
        """
        Sanity check: valid paths should pass validation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                )
            )
            
            errors = config.validate()
            
            # Should have no path-related errors
            path_errors = [e for e in errors if "does not exist" in e]
            assert len(path_errors) == 0


# =============================================================================
# Property 23: Invalid Configuration Error Messages
# **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
# **Validates: Requirements 10.5**
#
# For any invalid configuration, the system SHALL produce an error message
# indicating the specific issue.
# =============================================================================

class TestInvalidConfigurationErrorMessages:
    """Property tests for error message clarity (Property 23)."""

    @given(lr=invalid_learning_rate)
    @settings(max_examples=100)
    def test_invalid_learning_rate_produces_specific_error(self, lr: float):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any invalid learning rate, the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(learning_rate=lr)
            )
            
            errors = config.validate()
            
            # Should have an error about learning rate
            lr_errors = [e for e in errors if "learning rate" in e.lower()]
            assert len(lr_errors) > 0
            # Error should mention the actual value
            assert any(str(lr) in e for e in lr_errors)

    @given(batch_size=invalid_batch_size_negative)
    @settings(max_examples=100)
    def test_negative_batch_size_produces_specific_error(self, batch_size: int):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any negative batch size, the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(batch_size=batch_size)
            )
            
            errors = config.validate()
            
            # Should have an error about batch size
            batch_errors = [e for e in errors if "batch size" in e.lower()]
            assert len(batch_errors) > 0
            assert any(str(batch_size) in e for e in batch_errors)

    @given(batch_size=invalid_batch_size_not_divisible)
    @settings(max_examples=100)
    def test_batch_size_not_divisible_by_4_produces_specific_error(self, batch_size: int):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any batch size not divisible by 4, the error message SHALL indicate
        the quadrant strategy requirement.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(batch_size=batch_size)
            )
            
            errors = config.validate()
            
            # Should have an error about batch size divisibility
            divisibility_errors = [e for e in errors if "divisible by 4" in e.lower()]
            assert len(divisibility_errors) > 0
            assert any(str(batch_size) in e for e in divisibility_errors)

    @given(kl_coef=invalid_kl_coef)
    @settings(max_examples=100)
    def test_negative_kl_coef_produces_specific_error(self, kl_coef: float):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any negative KL coefficient, the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(kl_coef=kl_coef)
            )
            
            errors = config.validate()
            
            # Should have an error about KL coefficient
            kl_errors = [e for e in errors if "kl" in e.lower()]
            assert len(kl_errors) > 0
            assert any(str(kl_coef) in e for e in kl_errors)

    @given(gamma=invalid_gamma)
    @settings(max_examples=100)
    def test_invalid_gamma_produces_specific_error(self, gamma: float):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any gamma outside [0, 1], the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(gamma=gamma)
            )
            
            errors = config.validate()
            
            # Should have an error about gamma
            gamma_errors = [e for e in errors if "gamma" in e.lower()]
            assert len(gamma_errors) > 0
            assert any(str(gamma) in e for e in gamma_errors)

    @given(num_episodes=invalid_positive_int)
    @settings(max_examples=100)
    def test_invalid_num_episodes_produces_specific_error(self, num_episodes: int):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any non-positive number of episodes, the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                schedule=TrainingSchedule(num_episodes=num_episodes)
            )
            
            errors = config.validate()
            
            # Should have an error about number of episodes
            episode_errors = [e for e in errors if "episodes" in e.lower()]
            assert len(episode_errors) > 0
            assert any(str(num_episodes) in e for e in episode_errors)

    @given(num_gpus=invalid_positive_int)
    @settings(max_examples=100)
    def test_invalid_num_gpus_produces_specific_error(self, num_gpus: int):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        For any non-positive number of GPUs, the error message SHALL indicate the issue.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path=tmpdir),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                ),
                infrastructure=InfrastructureConfig(num_gpus=num_gpus)
            )
            
            errors = config.validate()
            
            # Should have an error about number of GPUs
            gpu_errors = [e for e in errors if "gpu" in e.lower()]
            assert len(gpu_errors) > 0
            assert any(str(num_gpus) in e for e in gpu_errors)

    def test_multiple_errors_all_reported(self):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        When multiple configuration issues exist, ALL issues SHALL be reported.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path="/nonexistent/model"),
                data_paths=DataPaths(
                    medec_data_path="/nonexistent/data",
                    output_dir=tmpdir
                ),
                hyperparameters=TrainingHyperparameters(
                    learning_rate=-1.0,
                    batch_size=5,  # Not divisible by 4
                    kl_coef=-0.5
                ),
                schedule=TrainingSchedule(num_episodes=0)
            )
            
            errors = config.validate()
            
            # Should report all issues
            assert len(errors) >= 4  # At least path, lr, batch_size, episodes
            
            # Check specific issues are mentioned
            error_text = " ".join(errors).lower()
            assert "does not exist" in error_text
            assert "learning rate" in error_text
            assert "batch size" in error_text or "divisible" in error_text

    def test_error_message_format_in_exception(self):
        """
        **Feature: medserl-adaptation, Property 23: Invalid Configuration Error Messages**
        **Validates: Requirements 10.5**
        
        ConfigurationError exception message SHALL clearly list all issues.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MedSERLConfig(
                model_paths=ModelPaths(base_model_path="/nonexistent/path"),
                data_paths=DataPaths(
                    medec_data_path=tmpdir,
                    output_dir=tmpdir
                )
            )
            
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate_or_raise()
            
            error_message = str(exc_info.value)
            
            # Error message should be informative
            assert "validation failed" in error_message.lower()
            assert "/nonexistent/path" in error_message
            assert "does not exist" in error_message
