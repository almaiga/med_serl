"""
Property-based tests for MedSeRL data processor.

**Feature: medserl-adaptation, Property 1: Error Pool Partitioning**
**Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
"""

from typing import Dict, List, Optional

from hypothesis import given, strategies as st, settings

from src.data_processor import MedicalDataProcessor


# =============================================================================
# Strategies for generating MEDEC-like entries
# =============================================================================

# Valid MEDEC error types
VALID_ERROR_TYPES = [
    'Diagnosis', 'Management', 'Treatment',
    'Pharmacotherapy', 'Causal Organism'
]

# Strategy for generating clinical note text
clinical_text_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' .,;:-'
    ),
    min_size=10,
    max_size=200
).filter(lambda s: s.strip())

# Strategy for generating text IDs
text_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N')),
    min_size=1,
    max_size=20
)

# Strategy for generating error types
error_type_strategy = st.sampled_from(VALID_ERROR_TYPES)

# Strategy for generating optional corrected text
optional_corrected_text_strategy = st.one_of(
    st.none(),
    clinical_text_strategy
)


def create_medec_entry(
    text: str,
    text_id: str,
    error_flag: int,
    error_type: Optional[str] = None,
    corrected_text: Optional[str] = None,
    subset: str = 'MS',
    source_file: str = 'test.csv'
) -> Dict:
    """Create a MEDEC-like entry dictionary."""
    return {
        'text_id': text_id,
        'text': text,
        'sentences': text,
        'error_flag': error_flag,
        'error_type': error_type,
        'error_sentence_id': 0 if error_flag == 1 else -1,
        'error_sentence': text[:50] if error_flag == 1 else None,
        'corrected_sentence': corrected_text[:50] if corrected_text else None,
        'corrected_text': corrected_text,
        'subset': subset,
        'source_file': source_file
    }


# =============================================================================
# Property 1: Error Pool Partitioning
# **Feature: medserl-adaptation, Property 1: Error Pool Partitioning**
# **Validates: Requirements 1.2**
#
# For any MEDEC entry with error_flag equal to 1, the entry SHALL be added
# to the error pool with its associated error type.
# =============================================================================

class TestErrorPoolPartitioning:
    """Property tests for error pool partitioning (Property 1)."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_error_flag_1_entries_added_to_error_pool(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 1: Error Pool Partitioning**
        **Validates: Requirements 1.2**

        For any MEDEC entry with error_flag equal to 1, the entry SHALL be
        added to the error pool with its associated error type.
        """
        # Create an entry with error_flag=1
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=1,
            error_type=error_type
        )

        # Create a processor and manually set raw_data
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        # Run partitioning
        processor._partition_data()

        # Verify entry is in error pool
        assert len(processor.error_pool) == 1
        error_entry = processor.error_pool[0]
        assert error_entry['text'] == text
        assert error_entry['text_id'] == text_id
        assert error_entry['error_type'] == error_type
        assert error_entry['label'] == 'Error'

    @given(
        texts=st.lists(clinical_text_strategy, min_size=1, max_size=10),
        error_types=st.lists(error_type_strategy, min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_all_error_entries_in_error_pool(
        self,
        texts: List[str],
        error_types: List[str]
    ):
        """
        **Feature: medserl-adaptation, Property 1: Error Pool Partitioning**
        **Validates: Requirements 1.2**

        For any collection of MEDEC entries with error_flag=1, ALL entries
        SHALL be added to the error pool.
        """
        # Create multiple error entries
        entries = []
        for i, (text, error_type) in enumerate(zip(texts, error_types)):
            entry = create_medec_entry(
                text=text,
                text_id=f"test_{i}",
                error_flag=1,
                error_type=error_type
            )
            entries.append(entry)

        # Create processor and partition
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = entries
        processor.subsets = ['MS']

        processor._partition_data()

        # Verify all entries are in error pool
        assert len(processor.error_pool) == len(entries)

        # Verify each entry's error type is preserved
        for i, error_entry in enumerate(processor.error_pool):
            original = entries[i]
            assert error_entry['error_type'] == original['error_type']

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_error_pool_entry_has_required_fields(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 1: Error Pool Partitioning**
        **Validates: Requirements 1.2**

        For any entry added to the error pool, it SHALL contain the required
        fields: text, text_id, error_type, and label.
        """
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=1,
            error_type=error_type
        )

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        processor._partition_data()

        assert len(processor.error_pool) == 1
        error_entry = processor.error_pool[0]

        # Verify required fields exist
        assert 'text' in error_entry
        assert 'text_id' in error_entry
        assert 'error_type' in error_entry
        assert 'label' in error_entry
        assert error_entry['label'] == 'Error'


# =============================================================================
# Property 2: Clean Pool Partitioning
# **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
# **Validates: Requirements 1.3**
#
# For any MEDEC entry with error_flag equal to 0 OR with a non-empty
# corrected_text field, the entry SHALL be added to the clean pool.
# =============================================================================

class TestCleanPoolPartitioning:
    """Property tests for clean pool partitioning (Property 2)."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_error_flag_0_entries_added_to_clean_pool(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
        **Validates: Requirements 1.3**

        For any MEDEC entry with error_flag equal to 0, the entry SHALL be
        added to the clean pool.
        """
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=0
        )

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        processor._partition_data()

        # Verify entry is in clean pool
        assert len(processor.clean_pool) == 1
        clean_entry = processor.clean_pool[0]
        assert clean_entry['text'] == text
        assert clean_entry['text_id'] == text_id
        assert clean_entry['label'] == 'Clean'

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy,
        corrected_text=clinical_text_strategy
    )
    @settings(max_examples=100)
    def test_corrected_text_added_to_clean_pool(
        self,
        text: str,
        text_id: str,
        error_type: str,
        corrected_text: str
    ):
        """
        **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
        **Validates: Requirements 1.3**

        For any MEDEC entry with a non-empty corrected_text field, the
        corrected_text SHALL be added to the clean pool.
        """
        # Entry with error_flag=1 but has corrected_text
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=1,
            error_type=error_type,
            corrected_text=corrected_text
        )

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        processor._partition_data()

        # Verify corrected_text is in clean pool
        # (error entry goes to error pool, corrected goes to clean)
        assert len(processor.error_pool) == 1
        assert len(processor.clean_pool) == 1

        clean_entry = processor.clean_pool[0]
        assert clean_entry['text'] == corrected_text
        assert clean_entry['label'] == 'Clean'
        assert clean_entry.get('is_corrected') is True

    @given(
        texts=st.lists(clinical_text_strategy, min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_all_clean_entries_in_clean_pool(
        self,
        texts: List[str]
    ):
        """
        **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
        **Validates: Requirements 1.3**

        For any collection of MEDEC entries with error_flag=0, ALL entries
        SHALL be added to the clean pool.
        """
        entries = []
        for i, text in enumerate(texts):
            entry = create_medec_entry(
                text=text,
                text_id=f"test_{i}",
                error_flag=0
            )
            entries.append(entry)

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = entries
        processor.subsets = ['MS']

        processor._partition_data()

        # Verify all entries are in clean pool
        assert len(processor.clean_pool) == len(entries)

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_clean_pool_entry_has_required_fields(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
        **Validates: Requirements 1.3**

        For any entry added to the clean pool, it SHALL contain the required
        fields: text, text_id, and label.
        """
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=0
        )

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        processor._partition_data()

        assert len(processor.clean_pool) == 1
        clean_entry = processor.clean_pool[0]

        # Verify required fields exist
        assert 'text' in clean_entry
        assert 'text_id' in clean_entry
        assert 'label' in clean_entry
        assert clean_entry['label'] == 'Clean'

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy,
        corrected_text=clinical_text_strategy
    )
    @settings(max_examples=100)
    def test_both_pools_populated_for_error_with_correction(
        self,
        text: str,
        text_id: str,
        error_type: str,
        corrected_text: str
    ):
        """
        **Feature: medserl-adaptation, Property 2: Clean Pool Partitioning**
        **Validates: Requirements 1.3**

        For any MEDEC entry with error_flag=1 AND corrected_text, the original
        SHALL go to error pool and corrected_text SHALL go to clean pool.
        """
        entry = create_medec_entry(
            text=text,
            text_id=text_id,
            error_flag=1,
            error_type=error_type,
            corrected_text=corrected_text
        )

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.raw_data = [entry]
        processor.subsets = ['MS']

        processor._partition_data()

        # Both pools should have entries
        assert len(processor.error_pool) == 1
        assert len(processor.clean_pool) == 1

        # Error pool has original text
        assert processor.error_pool[0]['text'] == text
        assert processor.error_pool[0]['error_type'] == error_type

        # Clean pool has corrected text
        assert processor.clean_pool[0]['text'] == corrected_text
        assert processor.clean_pool[0]['label'] == 'Clean'


# =============================================================================
# Property 3: Quadrant Batch Balance
# **Feature: medserl-adaptation, Property 3: Quadrant Batch Balance**
# **Validates: Requirements 2.1, 2.3, 2.5, 2.7**
#
# For any batch generated by the Scribe Agent with size N (where N is
# divisible by 4), the batch SHALL contain exactly N/4 samples from each
# of the four quadrant modes (Augmented Ground Truth, Augmented Safe,
# Synthetic Decoy, Synthetic Injection).
# =============================================================================

# Mode mapping from internal mode names to quadrant sources
MODE_TO_SOURCE = {
    'augment_error': 'augmented_ground_truth',
    'augment_safe': 'augmented_safe',
    'make_decoy': 'synthetic_decoy',
    'inject_new_error': 'synthetic_injection'
}

# All expected quadrant sources
QUADRANT_SOURCES = [
    'augmented_ground_truth',
    'augmented_safe',
    'synthetic_decoy',
    'synthetic_injection'
]


class TestQuadrantBatchBalance:
    """Property tests for quadrant batch balance (Property 3)."""

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_batch_contains_equal_quadrant_distribution(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 3: Quadrant Batch Balance**
        **Validates: Requirements 2.1, 2.3, 2.5, 2.7**

        For any batch size N (divisible by 4), the batch SHALL contain
        exactly N/4 samples from each quadrant mode.
        """
        batch_size = batch_multiplier * 4

        # Create processor with sufficient data in both pools
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool with enough entries for stratification
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'Error note {i} with {error_type}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Populate clean pool with enough entries
        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Generate batch
        batch = processor.get_quadrant_batch(batch_size)

        # Verify batch size
        assert len(batch) == batch_size

        # Count samples per quadrant source
        source_counts = {source: 0 for source in QUADRANT_SOURCES}
        for sample in batch:
            source = sample['meta'].get('source')
            if source in source_counts:
                source_counts[source] += 1

        # Verify equal distribution (N/4 per quadrant)
        expected_per_quadrant = batch_size // 4
        for source, count in source_counts.items():
            assert count == expected_per_quadrant, (
                f"Expected {expected_per_quadrant} samples for {source}, "
                f"got {count}"
            )

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_all_four_quadrants_present(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 3: Quadrant Batch Balance**
        **Validates: Requirements 2.1, 2.3, 2.5, 2.7**

        For any valid batch, all four quadrant modes SHALL be represented.
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate pools
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'Error note {i}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        batch = processor.get_quadrant_batch(batch_size)

        # Collect all sources present
        sources_present = set()
        for sample in batch:
            source = sample['meta'].get('source')
            sources_present.add(source)

        # Verify all four quadrants are present
        for expected_source in QUADRANT_SOURCES:
            assert expected_source in sources_present, (
                f"Missing quadrant source: {expected_source}"
            )

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_augmented_ground_truth_samples_from_error_pool(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 3: Quadrant Batch Balance**
        **Validates: Requirements 2.1**

        Augmented Ground Truth samples SHALL be derived from the error pool.
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Create distinct error pool entries
        error_texts = set()
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            text = f'ERROR_POOL_TEXT_{i}'
            error_texts.add(text)
            processor.error_pool.append({
                'text': text,
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Create distinct clean pool entries
        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'CLEAN_POOL_TEXT_{i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        batch = processor.get_quadrant_batch(batch_size)

        # Verify augmented_ground_truth samples come from error pool
        for sample in batch:
            if sample['meta'].get('source') == 'augmented_ground_truth':
                assert sample['original_text'] in error_texts, (
                    "Augmented Ground Truth sample not from error pool"
                )
                assert sample['meta'].get('has_error') is True

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_other_quadrants_sample_from_clean_pool(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 3: Quadrant Batch Balance**
        **Validates: Requirements 2.3, 2.5, 2.7**

        Augmented Safe, Synthetic Decoy, and Synthetic Injection samples
        SHALL be derived from the clean pool.
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Create error pool entries
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'ERROR_POOL_TEXT_{i}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Create distinct clean pool entries
        clean_texts = set()
        for i in range(batch_size * 3):
            text = f'CLEAN_POOL_TEXT_{i}'
            clean_texts.add(text)
            processor.clean_pool.append({
                'text': text,
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        batch = processor.get_quadrant_batch(batch_size)

        # Verify other quadrant samples come from clean pool
        clean_quadrants = ['augmented_safe', 'synthetic_decoy', 'synthetic_injection']
        for sample in batch:
            source = sample['meta'].get('source')
            if source in clean_quadrants:
                assert sample['original_text'] in clean_texts, (
                    f"{source} sample not from clean pool"
                )


# =============================================================================
# Property 4: Augmented Ground Truth Prompt Structure
# **Feature: medserl-adaptation, Property 4: Augmented Ground Truth Prompt**
# **Validates: Requirements 2.2**
#
# For any error note selected for Augmented Ground Truth mode, the generated
# prompt SHALL contain instructions to preserve the original error type while
# modifying demographics.
# =============================================================================

class TestAugmentedGroundTruthPromptStructure:
    """Property tests for Augmented Ground Truth prompt structure."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_preserve_error_instruction(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 4: Augmented Ground Truth**
        **Validates: Requirements 2.2**

        For any error note in Augmented Ground Truth mode, the prompt SHALL
        contain instructions to preserve/keep the original error.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = [{
            'text': text,
            'text_id': text_id,
            'error_type': error_type,
            'label': 'Error',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Generate augmented ground truth samples
        samples = processor._sample_augmented_ground_truth(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must contain instruction to preserve/keep the error
        assert 'keep' in prompt or 'preserve' in prompt, (
            "Prompt must instruct to keep/preserve the error"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_error_type(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 4: Augmented Ground Truth**
        **Validates: Requirements 2.2**

        For any error note in Augmented Ground Truth mode, the prompt SHALL
        reference the specific error type being preserved.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = [{
            'text': text,
            'text_id': text_id,
            'error_type': error_type,
            'label': 'Error',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_augmented_ground_truth(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must mention the error type
        assert error_type.lower() in prompt, (
            f"Prompt must mention the error type '{error_type}'"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_demographics_change_instruction(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 4: Augmented Ground Truth**
        **Validates: Requirements 2.2**

        For any error note in Augmented Ground Truth mode, the prompt SHALL
        contain instructions to modify demographics/patient details.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = [{
            'text': text,
            'text_id': text_id,
            'error_type': error_type,
            'label': 'Error',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_augmented_ground_truth(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must mention changing demographics or patient details
        has_demographics = 'demographic' in prompt
        has_patient = 'patient' in prompt
        has_change = 'change' in prompt or 'rewrite' in prompt

        assert has_demographics or (has_patient and has_change), (
            "Prompt must instruct to change demographics/patient details"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy,
        error_type=error_type_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_original_note(
        self,
        text: str,
        text_id: str,
        error_type: str
    ):
        """
        **Feature: medserl-adaptation, Property 4: Augmented Ground Truth**
        **Validates: Requirements 2.2**

        For any error note in Augmented Ground Truth mode, the prompt SHALL
        include the original clinical note text.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = [{
            'text': text,
            'text_id': text_id,
            'error_type': error_type,
            'label': 'Error',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_augmented_ground_truth(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt']

        # Prompt must contain the original note text
        assert text in prompt, (
            "Prompt must include the original clinical note text"
        )


# =============================================================================
# Property 5: Augmented Safe Prompt Structure
# **Feature: medserl-adaptation, Property 5: Augmented Safe Prompt Structure**
# **Validates: Requirements 2.4**
#
# For any clean note selected for Augmented Safe mode, the generated prompt
# SHALL contain instructions to paraphrase while maintaining medical accuracy.
# =============================================================================

class TestAugmentedSafePromptStructure:
    """Property tests for Augmented Safe prompt structure."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_paraphrase_instruction(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 5: Augmented Safe Prompt**
        **Validates: Requirements 2.4**

        For any clean note in Augmented Safe mode, the prompt SHALL contain
        instructions to paraphrase the note.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="augment_safe"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must contain paraphrase instruction
        assert 'paraphrase' in prompt, (
            "Prompt must instruct to paraphrase the note"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_accuracy_instruction(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 5: Augmented Safe Prompt**
        **Validates: Requirements 2.4**

        For any clean note in Augmented Safe mode, the prompt SHALL contain
        instructions to maintain medical accuracy.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="augment_safe"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must mention accuracy or error-free
        has_accurate = 'accurate' in prompt or 'accuracy' in prompt
        has_error_free = 'error-free' in prompt or 'error free' in prompt

        assert has_accurate or has_error_free, (
            "Prompt must instruct to maintain medical accuracy"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_original_note(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 5: Augmented Safe Prompt**
        **Validates: Requirements 2.4**

        For any clean note in Augmented Safe mode, the prompt SHALL include
        the original clinical note text.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="augment_safe"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt']

        # Prompt must contain the original note text
        assert text in prompt, (
            "Prompt must include the original clinical note text"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_metadata_indicates_no_error(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 5: Augmented Safe Prompt**
        **Validates: Requirements 2.4**

        For any Augmented Safe sample, the metadata SHALL indicate has_error
        is False (clean note remains clean after paraphrasing).
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="augment_safe"
        )

        assert len(samples) == 1
        meta = samples[0]['meta']

        assert meta.get('has_error') is False, (
            "Augmented Safe metadata must indicate has_error=False"
        )
        assert meta.get('source') == 'augmented_safe', (
            "Metadata source must be 'augmented_safe'"
        )


# =============================================================================
# Property 6: Synthetic Decoy Prompt Structure
# **Feature: medserl-adaptation, Property 6: Synthetic Decoy Prompt Structure**
# **Validates: Requirements 2.6**
#
# For any clean note selected for Synthetic Decoy mode, the generated prompt
# SHALL contain instructions to inject cosmetic noise without introducing
# medical errors.
# =============================================================================

class TestSyntheticDecoyPromptStructure:
    """Property tests for Synthetic Decoy prompt structure."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_cosmetic_noise_instruction(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 6: Synthetic Decoy Prompt**
        **Validates: Requirements 2.6**

        For any clean note in Synthetic Decoy mode, the prompt SHALL contain
        instructions to inject cosmetic noise (typos, formatting issues).
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="make_decoy"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must mention cosmetic noise, typos, or formatting
        has_cosmetic = 'cosmetic' in prompt
        has_typo = 'typo' in prompt
        has_formatting = 'formatting' in prompt
        has_noise = 'noise' in prompt

        assert has_cosmetic or has_typo or has_formatting or has_noise, (
            "Prompt must instruct to inject cosmetic noise/typos/formatting"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_no_medical_error_instruction(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 6: Synthetic Decoy Prompt**
        **Validates: Requirements 2.6**

        For any clean note in Synthetic Decoy mode, the prompt SHALL contain
        instructions to NOT introduce medical errors.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="make_decoy"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must contain instruction to avoid medical errors
        has_do_not = 'do not' in prompt or "don't" in prompt
        has_no_error = 'no' in prompt and 'error' in prompt
        has_medical = 'medical' in prompt

        assert (has_do_not and has_medical) or has_no_error, (
            "Prompt must instruct to NOT introduce medical errors"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_original_note(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 6: Synthetic Decoy Prompt**
        **Validates: Requirements 2.6**

        For any clean note in Synthetic Decoy mode, the prompt SHALL include
        the original clinical note text.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="make_decoy"
        )

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt']

        # Prompt must contain the original note text
        assert text in prompt, (
            "Prompt must include the original clinical note text"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_metadata_indicates_no_error(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 6: Synthetic Decoy Prompt**
        **Validates: Requirements 2.6**

        For any Synthetic Decoy sample, the metadata SHALL indicate has_error
        is False (cosmetic noise does not constitute a medical error).
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_and_process(
            processor.clean_pool, 1, mode="make_decoy"
        )

        assert len(samples) == 1
        meta = samples[0]['meta']

        assert meta.get('has_error') is False, (
            "Synthetic Decoy metadata must indicate has_error=False"
        )
        assert meta.get('source') == 'synthetic_decoy', (
            "Metadata source must be 'synthetic_decoy'"
        )


# =============================================================================
# Property 7: Synthetic Injection Prompt Structure
# **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
# **Validates: Requirements 2.8**
#
# For any clean note selected for Synthetic Injection mode, the generated
# prompt SHALL contain instructions to inject a new medical error.
# =============================================================================

class TestSyntheticInjectionPromptStructure:
    """Property tests for Synthetic Injection prompt structure."""

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_inject_error_instruction(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
        **Validates: Requirements 2.8**

        For any clean note in Synthetic Injection mode, the prompt SHALL
        contain instructions to inject/introduce a medical error.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_synthetic_injection(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must contain instruction to inject/include an error
        has_inject = 'inject' in prompt
        has_include = 'include' in prompt
        has_introduce = 'introduce' in prompt
        has_error = 'error' in prompt

        assert (has_inject or has_include or has_introduce) and has_error, (
            "Prompt must instruct to inject/include/introduce an error"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_error_type(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
        **Validates: Requirements 2.8**

        For any clean note in Synthetic Injection mode, the prompt SHALL
        specify a target error type to inject.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_synthetic_injection(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()

        # Prompt must mention one of the valid error types
        error_type_mentioned = any(
            et.lower() in prompt for et in VALID_ERROR_TYPES
        )

        assert error_type_mentioned, (
            "Prompt must specify a target error type from VALID_ERROR_TYPES"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_prompt_contains_original_note(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
        **Validates: Requirements 2.8**

        For any clean note in Synthetic Injection mode, the prompt SHALL
        include the original clinical note text.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_synthetic_injection(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt']

        # Prompt must contain the original note text
        assert text in prompt, (
            "Prompt must include the original clinical note text"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_metadata_indicates_error(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
        **Validates: Requirements 2.8**

        For any Synthetic Injection sample, the metadata SHALL indicate
        has_error is True and include the target error type.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_synthetic_injection(1)

        assert len(samples) == 1
        meta = samples[0]['meta']

        assert meta.get('has_error') is True, (
            "Synthetic Injection metadata must indicate has_error=True"
        )
        assert meta.get('source') == 'synthetic_injection', (
            "Metadata source must be 'synthetic_injection'"
        )
        assert meta.get('error_type') in VALID_ERROR_TYPES, (
            "Metadata must include a valid error_type"
        )

    @given(
        text=clinical_text_strategy,
        text_id=text_id_strategy
    )
    @settings(max_examples=100)
    def test_metadata_error_type_matches_prompt(
        self,
        text: str,
        text_id: str
    ):
        """
        **Feature: medserl-adaptation, Property 7: Synthetic Injection Prompt**
        **Validates: Requirements 2.8**

        For any Synthetic Injection sample, the error type in metadata SHALL
        match the error type mentioned in the prompt.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = [{
            'text': text,
            'text_id': text_id,
            'label': 'Clean',
            'subset': 'MS',
            'source_file': 'test.csv'
        }]
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        samples = processor._sample_synthetic_injection(1)

        assert len(samples) == 1
        prompt = samples[0]['scribe_prompt'].lower()
        meta_error_type = samples[0]['meta'].get('error_type')

        # The error type in metadata should appear in the prompt
        assert meta_error_type.lower() in prompt, (
            f"Metadata error_type '{meta_error_type}' must appear in prompt"
        )



# =============================================================================
# Property 8: Batch Randomization
# **Feature: medserl-adaptation, Property 8: Batch Randomization**
# **Validates: Requirements 2.9**
#
# For any generated batch, the final order of samples SHALL differ from the
# sequential quadrant generation order (i.e., samples are shuffled).
# =============================================================================

class TestBatchRandomization:
    """Property tests for batch randomization (Property 8)."""

    @given(
        batch_multiplier=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=100)
    def test_batch_order_differs_from_sequential_generation(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 8: Batch Randomization**
        **Validates: Requirements 2.9**

        For any generated batch, the final order of samples SHALL differ
        from the sequential quadrant generation order.
        
        Note: We use batch_multiplier >= 2 (batch_size >= 8) to minimize
        the probability of random shuffle matching original order by chance.
        For batch_size=8, probability of matching is 1/40320 ≈ 0.0025%.
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool with enough entries for stratification
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'Error note {i} with {error_type}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Populate clean pool with enough entries
        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Generate batch
        batch = processor.get_quadrant_batch(batch_size)

        # Extract the source order from the batch
        sources = [sample['meta'].get('source') for sample in batch]

        # The sequential generation order would be:
        # [augmented_ground_truth] * quarter + [augmented_safe] * quarter +
        # [synthetic_decoy] * quarter + [synthetic_injection] * quarter
        quarter = batch_size // 4
        sequential_order = (
            ['augmented_ground_truth'] * quarter +
            ['augmented_safe'] * quarter +
            ['synthetic_decoy'] * quarter +
            ['synthetic_injection'] * quarter
        )

        # The batch should be shuffled, so the order should differ
        # from the sequential generation order
        assert sources != sequential_order, (
            "Batch order should differ from sequential quadrant generation order. "
            "The batch should be shuffled after generation."
        )

    @given(
        batch_multiplier=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100)
    def test_multiple_batches_have_different_orders(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 8: Batch Randomization**
        **Validates: Requirements 2.9**

        For any two consecutively generated batches, the sample orders
        SHALL differ (demonstrating randomization is applied).
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate pools with enough entries
        for i in range(batch_size * 4):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'Error note {i} with {error_type}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        for i in range(batch_size * 6):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Generate two batches
        batch1 = processor.get_quadrant_batch(batch_size)
        batch2 = processor.get_quadrant_batch(batch_size)

        # Extract source orders
        sources1 = [sample['meta'].get('source') for sample in batch1]
        sources2 = [sample['meta'].get('source') for sample in batch2]

        # The two batches should have different orders
        # (probability of same order is extremely low for batch_size >= 8)
        assert sources1 != sources2, (
            "Two consecutively generated batches should have different orders "
            "due to randomization."
        )

    @given(
        batch_multiplier=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100)
    def test_shuffling_preserves_all_samples(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 8: Batch Randomization**
        **Validates: Requirements 2.9**

        For any generated batch, shuffling SHALL preserve all samples
        (no samples lost or duplicated during randomization).
        """
        batch_size = batch_multiplier * 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate pools
        for i in range(batch_size * 2):
            error_type = VALID_ERROR_TYPES[i % len(VALID_ERROR_TYPES)]
            processor.error_pool.append({
                'text': f'Error note {i} with {error_type}',
                'text_id': f'error_{i}',
                'error_type': error_type,
                'label': 'Error',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Generate batch
        batch = processor.get_quadrant_batch(batch_size)

        # Verify batch size is preserved
        assert len(batch) == batch_size, (
            f"Batch size should be {batch_size}, got {len(batch)}"
        )

        # Count samples per quadrant source
        source_counts = {source: 0 for source in QUADRANT_SOURCES}
        for sample in batch:
            source = sample['meta'].get('source')
            if source in source_counts:
                source_counts[source] += 1

        # Verify equal distribution is preserved after shuffling
        expected_per_quadrant = batch_size // 4
        for source, count in source_counts.items():
            assert count == expected_per_quadrant, (
                f"Expected {expected_per_quadrant} samples for {source} after "
                f"shuffling, got {count}. Shuffling should preserve sample counts."
            )


# =============================================================================
# Property 24: Augmented Ground Truth Error Type Stratification
# **Feature: medserl-adaptation, Property 24: Augmented Ground Truth Error Type Stratification**
# **Validates: Requirements 2.10**
#
# For any batch with Augmented Ground Truth samples, the samples SHALL be
# stratified across all five error types (Diagnosis, Management, Treatment,
# Pharmacotherapy, Causal Organism) when sufficient samples exist in the
# error pool.
# =============================================================================

class TestAugmentedGroundTruthErrorTypeStratification:
    """Property tests for Augmented Ground Truth error type stratification."""

    @given(
        sample_count=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100)
    def test_stratification_covers_all_error_types(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 24: Augmented Ground Truth Error Type Stratification**
        **Validates: Requirements 2.10**

        For any batch with Augmented Ground Truth samples where sufficient
        samples exist in the error pool, the samples SHALL include
        representation from all five error types.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool with multiple entries per error type
        for error_type in VALID_ERROR_TYPES:
            for i in range(sample_count):
                processor.error_pool.append({
                    'text': f'Error note {i} with {error_type}',
                    'text_id': f'{error_type}_{i}',
                    'error_type': error_type,
                    'label': 'Error',
                    'subset': 'MS',
                    'source_file': 'test.csv'
                })

        # Request enough samples to cover all error types
        samples = processor._sample_augmented_ground_truth(sample_count)

        # Collect error types from samples
        sampled_error_types = set()
        for sample in samples:
            error_type = sample['meta'].get('error_type')
            if error_type:
                sampled_error_types.add(error_type)

        # When sample_count >= 5, all error types should be represented
        assert len(sampled_error_types) == len(VALID_ERROR_TYPES), (
            f"Expected all {len(VALID_ERROR_TYPES)} error types to be "
            f"represented, got {len(sampled_error_types)}: {sampled_error_types}"
        )

    @given(
        sample_count=st.integers(min_value=5, max_value=25)
    )
    @settings(max_examples=100)
    def test_stratification_balances_error_types(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 24: Augmented Ground Truth Error Type Stratification**
        **Validates: Requirements 2.10**

        For any batch with Augmented Ground Truth samples, the error types
        SHALL be approximately balanced (within 1 sample difference due to
        remainder distribution).
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool with many entries per error type
        for error_type in VALID_ERROR_TYPES:
            for i in range(sample_count * 2):
                processor.error_pool.append({
                    'text': f'Error note {i} with {error_type}',
                    'text_id': f'{error_type}_{i}',
                    'error_type': error_type,
                    'label': 'Error',
                    'subset': 'MS',
                    'source_file': 'test.csv'
                })

        samples = processor._sample_augmented_ground_truth(sample_count)

        # Count samples per error type
        error_type_counts = {et: 0 for et in VALID_ERROR_TYPES}
        for sample in samples:
            error_type = sample['meta'].get('error_type')
            if error_type in error_type_counts:
                error_type_counts[error_type] += 1

        # Calculate expected distribution
        expected_per_type = sample_count // len(VALID_ERROR_TYPES)
        remainder = sample_count % len(VALID_ERROR_TYPES)

        # Each error type should have expected_per_type or expected_per_type + 1
        for error_type, count in error_type_counts.items():
            assert expected_per_type <= count <= expected_per_type + 1, (
                f"Error type '{error_type}' has {count} samples, expected "
                f"between {expected_per_type} and {expected_per_type + 1}"
            )

        # Total should match requested count
        total = sum(error_type_counts.values())
        assert total == sample_count, (
            f"Total samples {total} should equal requested {sample_count}"
        )

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_quadrant_batch_uses_stratified_augmented_ground_truth(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 24: Augmented Ground Truth Error Type Stratification**
        **Validates: Requirements 2.10**

        For any batch generated via get_quadrant_batch, the Augmented Ground
        Truth samples SHALL be stratified across error types.
        """
        batch_size = batch_multiplier * 4
        quarter = batch_size // 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool with entries for each error type
        for error_type in VALID_ERROR_TYPES:
            for i in range(batch_size):
                processor.error_pool.append({
                    'text': f'Error note {i} with {error_type}',
                    'text_id': f'{error_type}_{i}',
                    'error_type': error_type,
                    'label': 'Error',
                    'subset': 'MS',
                    'source_file': 'test.csv'
                })

        # Populate clean pool
        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        batch = processor.get_quadrant_batch(batch_size)

        # Extract augmented_ground_truth samples
        agt_samples = [
            s for s in batch
            if s['meta'].get('source') == 'augmented_ground_truth'
        ]

        assert len(agt_samples) == quarter

        # Count error types in AGT samples
        error_type_counts = {et: 0 for et in VALID_ERROR_TYPES}
        for sample in agt_samples:
            error_type = sample['meta'].get('error_type')
            if error_type in error_type_counts:
                error_type_counts[error_type] += 1

        # When quarter >= 5, all error types should be represented
        if quarter >= len(VALID_ERROR_TYPES):
            represented_types = [et for et, c in error_type_counts.items() if c > 0]
            assert len(represented_types) == len(VALID_ERROR_TYPES), (
                f"Expected all error types represented when quarter={quarter}, "
                f"got {len(represented_types)}: {represented_types}"
            )


# =============================================================================
# Property 25: Synthetic Injection Error Type Stratification
# **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
# **Validates: Requirements 2.11**
#
# For any batch with Synthetic Injection samples, the injected error types
# SHALL be stratified across all five error types to ensure balanced
# representation.
# =============================================================================

class TestSyntheticInjectionErrorTypeStratification:
    """Property tests for Synthetic Injection error type stratification."""

    @given(
        sample_count=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100)
    def test_stratification_covers_all_error_types(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
        **Validates: Requirements 2.11**

        For any batch with Synthetic Injection samples, the injected error
        types SHALL include representation from all five error types.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate clean pool with enough entries
        for i in range(sample_count * 2):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        # Generate synthetic injection samples
        samples = processor._sample_synthetic_injection(sample_count)

        # Collect injected error types
        injected_error_types = set()
        for sample in samples:
            error_type = sample['meta'].get('error_type')
            if error_type:
                injected_error_types.add(error_type)

        # When sample_count >= 5, all error types should be represented
        assert len(injected_error_types) == len(VALID_ERROR_TYPES), (
            f"Expected all {len(VALID_ERROR_TYPES)} error types to be "
            f"injected, got {len(injected_error_types)}: {injected_error_types}"
        )

    @given(
        sample_count=st.integers(min_value=5, max_value=25)
    )
    @settings(max_examples=100)
    def test_stratification_balances_injected_error_types(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
        **Validates: Requirements 2.11**

        For any batch with Synthetic Injection samples, the injected error
        types SHALL be approximately balanced (within 1 sample difference
        due to remainder distribution).
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate clean pool
        for i in range(sample_count * 2):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        samples = processor._sample_synthetic_injection(sample_count)

        # Count injected error types
        error_type_counts = {et: 0 for et in VALID_ERROR_TYPES}
        for sample in samples:
            error_type = sample['meta'].get('error_type')
            if error_type in error_type_counts:
                error_type_counts[error_type] += 1

        # Calculate expected distribution
        expected_per_type = sample_count // len(VALID_ERROR_TYPES)

        # Each error type should have expected_per_type or expected_per_type + 1
        for error_type, count in error_type_counts.items():
            assert expected_per_type <= count <= expected_per_type + 1, (
                f"Injected error type '{error_type}' has {count} samples, "
                f"expected between {expected_per_type} and {expected_per_type + 1}"
            )

        # Total should match requested count
        total = sum(error_type_counts.values())
        assert total == sample_count, (
            f"Total samples {total} should equal requested {sample_count}"
        )

    @given(
        batch_multiplier=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=100)
    def test_quadrant_batch_uses_stratified_synthetic_injection(
        self,
        batch_multiplier: int
    ):
        """
        **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
        **Validates: Requirements 2.11**

        For any batch generated via get_quadrant_batch, the Synthetic
        Injection samples SHALL have stratified injected error types.
        """
        batch_size = batch_multiplier * 4
        quarter = batch_size // 4

        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.error_pool = []
        processor.clean_pool = []
        processor.subsets = ['MS']
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        # Populate error pool
        for error_type in VALID_ERROR_TYPES:
            for i in range(batch_size):
                processor.error_pool.append({
                    'text': f'Error note {i} with {error_type}',
                    'text_id': f'{error_type}_{i}',
                    'error_type': error_type,
                    'label': 'Error',
                    'subset': 'MS',
                    'source_file': 'test.csv'
                })

        # Populate clean pool
        for i in range(batch_size * 3):
            processor.clean_pool.append({
                'text': f'Clean note {i}',
                'text_id': f'clean_{i}',
                'label': 'Clean',
                'subset': 'MS',
                'source_file': 'test.csv'
            })

        batch = processor.get_quadrant_batch(batch_size)

        # Extract synthetic_injection samples
        si_samples = [
            s for s in batch
            if s['meta'].get('source') == 'synthetic_injection'
        ]

        assert len(si_samples) == quarter

        # Count injected error types
        error_type_counts = {et: 0 for et in VALID_ERROR_TYPES}
        for sample in si_samples:
            error_type = sample['meta'].get('error_type')
            if error_type in error_type_counts:
                error_type_counts[error_type] += 1

        # When quarter >= 5, all error types should be represented
        if quarter >= len(VALID_ERROR_TYPES):
            represented_types = [et for et, c in error_type_counts.items() if c > 0]
            assert len(represented_types) == len(VALID_ERROR_TYPES), (
                f"Expected all error types in synthetic injection when "
                f"quarter={quarter}, got {len(represented_types)}: {represented_types}"
            )

    @given(
        sample_count=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=100)
    def test_get_stratified_error_types_returns_correct_count(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
        **Validates: Requirements 2.11**

        For any requested count, _get_stratified_error_types SHALL return
        exactly that many error types.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        error_types = processor._get_stratified_error_types(sample_count)

        assert len(error_types) == sample_count, (
            f"Expected {sample_count} error types, got {len(error_types)}"
        )

        # All returned types should be valid
        for error_type in error_types:
            assert error_type in VALID_ERROR_TYPES, (
                f"Invalid error type '{error_type}' returned"
            )

    @given(
        sample_count=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=100)
    def test_get_stratified_error_types_balanced_distribution(
        self,
        sample_count: int
    ):
        """
        **Feature: medserl-adaptation, Property 25: Synthetic Injection Error Type Stratification**
        **Validates: Requirements 2.11**

        For any requested count, _get_stratified_error_types SHALL return
        a balanced distribution of error types.
        """
        processor = MedicalDataProcessor.__new__(MedicalDataProcessor)
        processor.VALID_ERROR_TYPES = VALID_ERROR_TYPES

        error_types = processor._get_stratified_error_types(sample_count)

        # Count occurrences of each type
        type_counts = {et: 0 for et in VALID_ERROR_TYPES}
        for et in error_types:
            type_counts[et] += 1

        # Calculate expected distribution
        expected_per_type = sample_count // len(VALID_ERROR_TYPES)

        # Each type should have expected_per_type or expected_per_type + 1
        for error_type, count in type_counts.items():
            assert expected_per_type <= count <= expected_per_type + 1, (
                f"Error type '{error_type}' has {count} occurrences, "
                f"expected between {expected_per_type} and {expected_per_type + 1}"
            )
