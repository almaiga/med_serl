"""
MedicalDataProcessor: Loads and preprocesses MEDEC dataset for MedSeRL training.

Supports loading from:
- Local CSV files (data_raw/MEDEC/)
- HuggingFace datasets (abachaa/MEDEC)
"""

import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Optional


class MedicalDataProcessor:
    """
    Processes MEDEC dataset for medical error detection training.
    
    Partitions data into error and clean pools, and generates
    balanced batches using the 4-Quadrant Strategy.
    """
    
    # Column name mapping from CSV headers to internal names
    CSV_COLUMN_MAP = {
        'Text ID': 'text_id',
        'Text': 'text',
        'Sentences': 'sentences',
        'Error Flag': 'error_flag',
        'Error Type': 'error_type',
        'Error Sentence ID': 'error_sentence_id',
        'Error Sentence': 'error_sentence',
        'Corrected Sentence': 'corrected_sentence',
        'Corrected Text': 'corrected_text'
    }
    
    # Valid MEDEC error types
    VALID_ERROR_TYPES = [
        'Diagnosis', 'Management', 'Treatment', 
        'Pharmacotherapy', 'Causal Organism'
    ]
    
    # Files to exclude from training (reserved for testing)
    TEST_FILES = [
        'MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv',
        'MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv'
    ]

    def __init__(
        self,
        data_path: str = "data_raw/MEDEC",
        use_huggingface: bool = False,
        subsets: Optional[List[str]] = None,
        exclude_test_files: bool = True,
        load_test_only: bool = False
    ):
        """
        Initialize the processor by loading MEDEC and splitting into pools.

        Args:
            data_path: Path to local MEDEC data or HuggingFace dataset name
            use_huggingface: If True, load from HuggingFace; else local CSV
            subsets: List of subsets to load (e.g., ['MS', 'UW']).
                     If None, loads all available subsets.
            exclude_test_files: If True, exclude test files from training pools
            load_test_only: If True, load ONLY test files (for evaluation)
        """
        self.data_path = data_path
        self.use_huggingface = use_huggingface
        self.subsets = subsets or ['MS', 'UW']
        self.exclude_test_files = exclude_test_files
        self.load_test_only = load_test_only

        self.error_pool: List[Dict] = []
        self.clean_pool: List[Dict] = []
        self.raw_data: List[Dict] = []

        # Load data
        if use_huggingface:
            self._load_from_huggingface()
        else:
            self._load_from_local_csv()

        # Partition into pools
        self._partition_data()

        # Report counts
        self._report_counts()

    def _load_from_local_csv(self) -> None:
        """
        Load MEDEC data from local CSV files.

        Parses CSV files from data_raw/MEDEC/MEDEC-MS/ and MEDEC-UW/
        directories, extracting text, error_flag, error_type, and
        corrected_text fields.

        Test files (MS-TestSet, UW-TestSet) are excluded by default
        to preserve them for final model evaluation.
        """
        base_path = Path(self.data_path)

        if not base_path.exists():
            raise FileNotFoundError(f"MEDEC data path not found: {base_path}")

        loaded_files = []
        skipped_files = []

        for subset in self.subsets:
            subset_path = base_path / f"MEDEC-{subset}"

            if not subset_path.exists():
                print(f"Warning: Subset path not found: {subset_path}")
                continue

            # Find all CSV files in the subset directory
            csv_files = list(subset_path.glob("*.csv"))

            for csv_file in csv_files:
                # Check if this is a test file
                is_test_file = csv_file.name in self.TEST_FILES

                # Skip based on mode
                if self.load_test_only and not is_test_file:
                    skipped_files.append(csv_file.name)
                    continue
                if self.exclude_test_files and is_test_file:
                    skipped_files.append(csv_file.name)
                    continue

                try:
                    entries = self._parse_csv_file(csv_file, subset)
                    self.raw_data.extend(entries)
                    loaded_files.append(csv_file.name)
                except Exception as e:
                    print(f"Warning: Failed to load {csv_file}: {e}")

        if not self.raw_data:
            raise ValueError(f"No data loaded from {base_path}")

        mode = "test" if self.load_test_only else "training"
        print(f"Loaded {len(self.raw_data)} {mode} entries from {len(loaded_files)} files")
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files: {skipped_files}")
    
    def _parse_csv_file(self, csv_path: Path, subset: str) -> List[Dict]:
        """
        Parse a single CSV file and extract relevant fields.
        
        Args:
            csv_path: Path to the CSV file
            subset: Subset identifier (MS or UW)
            
        Returns:
            List of parsed entries with standardized field names
        """
        df = pd.read_csv(csv_path)
        
        # Rename columns to internal names
        df = df.rename(columns=self.CSV_COLUMN_MAP)
        
        entries = []
        for idx, row in df.iterrows():
            try:
                entry = self._parse_row(row, subset, csv_path.name)
                if entry:
                    entries.append(entry)
            except Exception as e:
                print(f"Warning: Skipping row {idx} in {csv_path.name}: {e}")
        
        return entries
    
    def _parse_row(self, row: pd.Series, subset: str, source_file: str) -> Optional[Dict]:
        """
        Parse a single row from the CSV file.
        
        Args:
            row: Pandas Series representing a row
            subset: Subset identifier
            source_file: Name of the source file
            
        Returns:
            Parsed entry dict or None if row is invalid
        """
        # Extract required fields
        text = row.get('text', '')
        if pd.isna(text) or not str(text).strip():
            return None
        
        # Parse error_flag (convert to int, handle NA)
        error_flag_raw = row.get('error_flag', 0)
        if pd.isna(error_flag_raw):
            error_flag = 0
        else:
            error_flag = int(error_flag_raw)
        
        # Parse error_type (handle NA values)
        error_type_raw = row.get('error_type', '')
        if pd.isna(error_type_raw) or str(error_type_raw).upper() == 'NA':
            error_type = None
        else:
            error_type = str(error_type_raw).strip()
        
        # Parse corrected_text (handle NA values)
        corrected_text_raw = row.get('corrected_text', '')
        if pd.isna(corrected_text_raw) or str(corrected_text_raw).upper() == 'NA':
            corrected_text = None
        else:
            corrected_text = str(corrected_text_raw).strip()
            if not corrected_text:
                corrected_text = None
        
        return {
            'text_id': str(row.get('text_id', '')),
            'text': str(text),
            'sentences': str(row.get('sentences', '')),
            'error_flag': error_flag,
            'error_type': error_type,
            'error_sentence_id': row.get('error_sentence_id', -1),
            'error_sentence': str(row.get('error_sentence', '')) if not pd.isna(row.get('error_sentence')) else None,
            'corrected_sentence': str(row.get('corrected_sentence', '')) if not pd.isna(row.get('corrected_sentence')) else None,
            'corrected_text': corrected_text,
            'subset': subset,
            'source_file': source_file
        }
    
    def _load_from_huggingface(self) -> None:
        """
        Load MEDEC data from HuggingFace datasets.
        """
        from datasets import load_dataset
        
        print(f"Loading MEDEC dataset from HuggingFace: {self.data_path}...")
        
        for subset in self.subsets:
            try:
                dataset = load_dataset(self.data_path, f"MEDEC-{subset}", split="train")
                for entry in dataset:
                    self.raw_data.append({
                        'text_id': entry.get('text_id', ''),
                        'text': entry.get('text', ''),
                        'sentences': entry.get('sentences', ''),
                        'error_flag': entry.get('error_flag', 0),
                        'error_type': entry.get('error_type'),
                        'error_sentence_id': entry.get('error_sentence_id', -1),
                        'error_sentence': entry.get('error_sentence'),
                        'corrected_sentence': entry.get('corrected_sentence'),
                        'corrected_text': entry.get('corrected_text'),
                        'subset': subset,
                        'source_file': 'huggingface'
                    })
            except Exception as e:
                print(f"Warning: Failed to load subset {subset} from HuggingFace: {e}")

    def _partition_data(self) -> None:
        """
        Partition loaded data into error and clean pools.
        
        - Notes with error_flag=1 go to error pool with their error type
        - Notes with error_flag=0 go to clean pool
        - Notes with corrected_text also contribute to clean pool
        
        Requirements: 1.2, 1.3
        """
        for entry in self.raw_data:
            # Add to error pool if error_flag == 1
            if entry['error_flag'] == 1:
                self.error_pool.append({
                    'text': entry['text'],
                    'text_id': entry['text_id'],
                    'error_type': entry['error_type'],
                    'label': 'Error',
                    'subset': entry['subset'],
                    'source_file': entry['source_file']
                })
            
            # Add to clean pool if error_flag == 0
            if entry['error_flag'] == 0:
                self.clean_pool.append({
                    'text': entry['text'],
                    'text_id': entry['text_id'],
                    'label': 'Clean',
                    'subset': entry['subset'],
                    'source_file': entry['source_file']
                })
            
            # Also add corrected_text to clean pool if available
            if entry.get('corrected_text'):
                self.clean_pool.append({
                    'text': entry['corrected_text'],
                    'text_id': f"{entry['text_id']}_corrected",
                    'label': 'Clean',
                    'subset': entry['subset'],
                    'source_file': entry['source_file'],
                    'is_corrected': True
                })
    
    def _report_counts(self) -> None:
        """
        Report the count of error notes and clean notes.
        
        Requirements: 1.4
        """
        print(f"Data Loaded: {len(self.error_pool)} Error Notes, {len(self.clean_pool)} Clean Notes.")
        
        # Report per-subset counts
        for subset in self.subsets:
            error_count = sum(1 for e in self.error_pool if e.get('subset') == subset)
            clean_count = sum(1 for e in self.clean_pool if e.get('subset') == subset)
            print(f"  {subset}: {error_count} errors, {clean_count} clean")
        
        # Report error type distribution
        error_types = {}
        for entry in self.error_pool:
            et = entry.get('error_type', 'Unknown')
            error_types[et] = error_types.get(et, 0) + 1
        
        if error_types:
            print("Error type distribution:")
            for et, count in sorted(error_types.items()):
                print(f"  {et}: {count}")

    def _normalize_error_type(self, error_type: str) -> Optional[str]:
        """
        Normalize an error type string to match VALID_ERROR_TYPES.
        
        Handles case variations and camelCase (e.g., 'causalOrganism' -> 'Causal Organism').
        
        Args:
            error_type: Raw error type string from data
            
        Returns:
            Normalized error type matching VALID_ERROR_TYPES, or None if no match
        """
        if not error_type:
            return None
        
        # Direct match
        if error_type in self.VALID_ERROR_TYPES:
            return error_type
        
        # Normalize: lowercase and remove spaces for comparison
        normalized = error_type.lower().replace(' ', '').replace('_', '')
        
        for valid_type in self.VALID_ERROR_TYPES:
            valid_normalized = valid_type.lower().replace(' ', '')
            if normalized == valid_normalized:
                return valid_type
        
        return None

    def _get_error_pool_by_type(self) -> Dict[str, List[Dict]]:
        """
        Group error pool entries by their error type.
        
        Returns:
            Dictionary mapping error type to list of entries with that type
        """
        error_by_type: Dict[str, List[Dict]] = {et: [] for et in self.VALID_ERROR_TYPES}
        
        for entry in self.error_pool:
            error_type = entry.get('error_type')
            normalized_type = self._normalize_error_type(error_type)
            
            if normalized_type:
                error_by_type[normalized_type].append(entry)
        
        return error_by_type

    def _sample_stratified_from_error_pool(self, count: int) -> List[Dict]:
        """
        Sample from error pool with stratification across all 5 error types.
        
        Ensures balanced representation of error types when sufficient samples exist.
        Uses round-robin allocation across error types.
        
        Args:
            count: Number of samples to select
            
        Returns:
            List of stratified samples from error pool
            
        Requirements: 2.10
        """
        error_by_type = self._get_error_pool_by_type()
        
        # Filter to types that have samples
        available_types = [et for et in self.VALID_ERROR_TYPES if error_by_type[et]]
        
        if not available_types:
            return []
        
        # Calculate samples per type (round-robin allocation)
        samples_per_type = count // len(available_types)
        remainder = count % len(available_types)
        
        stratified_samples = []
        
        # Shuffle available types to randomize which types get extra samples
        shuffled_types = available_types.copy()
        random.shuffle(shuffled_types)
        
        for i, error_type in enumerate(shuffled_types):
            pool_for_type = error_by_type[error_type]
            
            # Allocate extra sample to first 'remainder' types
            type_count = samples_per_type + (1 if i < remainder else 0)
            
            # Sample from this error type's pool
            actual_count = min(type_count, len(pool_for_type))
            if actual_count > 0:
                type_samples = random.sample(pool_for_type, actual_count)
                stratified_samples.extend(type_samples)
        
        return stratified_samples

    def _get_stratified_error_types(self, count: int) -> List[str]:
        """
        Generate a stratified list of error types for synthetic injection.
        
        Ensures balanced representation across all 5 error type categories.
        
        Args:
            count: Number of error types to generate
            
        Returns:
            List of error types, stratified across all 5 categories
            
        Requirements: 2.11
        """
        # Calculate how many of each type we need
        types_per_category = count // len(self.VALID_ERROR_TYPES)
        remainder = count % len(self.VALID_ERROR_TYPES)
        
        stratified_types = []
        
        # Shuffle to randomize which types get extra allocation
        shuffled_types = self.VALID_ERROR_TYPES.copy()
        random.shuffle(shuffled_types)
        
        for i, error_type in enumerate(shuffled_types):
            # Allocate extra to first 'remainder' types
            type_count = types_per_category + (1 if i < remainder else 0)
            stratified_types.extend([error_type] * type_count)
        
        # Shuffle the final list to avoid predictable ordering
        random.shuffle(stratified_types)
        
        return stratified_types

    def get_quadrant_batch(self, batch_size: int = 4) -> List[Dict]:
        """
        Generate a balanced batch according to the 4-Quadrant Strategy.
        
        batch_size must be divisible by 4.
        
        Args:
            batch_size: Total batch size (must be divisible by 4)
            
        Returns:
            List of samples with scribe prompts and metadata
        """
        if batch_size % 4 != 0:
            raise ValueError(f"batch_size must be divisible by 4, got {batch_size}")
        
        quarter = batch_size // 4
        batch = []

        # Q1: Augmented Ground Truth (Real Error)
        # Goal: Generalization (Don't memorize text)
        # Uses stratified sampling across error types (Requirement 2.10)
        batch.extend(self._sample_augmented_ground_truth(quarter))

        # Q2: Augmented Safe (Real Clean)
        # Goal: Stability (Don't become paranoid)
        batch.extend(self._sample_and_process(
            self.clean_pool, quarter, mode="augment_safe"
        ))

        # Q3: Synthetic Decoy (Clean -> Noisy)
        # Goal: Robustness (Ignore typos)
        batch.extend(self._sample_and_process(
            self.clean_pool, quarter, mode="make_decoy"
        ))

        # Q4: Synthetic Injection (Clean -> New Error)
        # Goal: Expansion (New error types)
        # Uses stratified error type injection (Requirement 2.11)
        batch.extend(self._sample_synthetic_injection(quarter))

        random.shuffle(batch)
        return batch

    def _sample_augmented_ground_truth(self, count: int) -> List[Dict]:
        """
        Sample for Augmented Ground Truth mode with error type stratification.
        
        Stratifies sampling across all 5 error types to ensure balanced
        representation when sufficient samples exist in the error pool.
        
        Args:
            count: Number of samples to generate
            
        Returns:
            List of processed samples for augmented ground truth mode
            
        Requirements: 2.1, 2.2, 2.10
        """
        # Get stratified samples from error pool
        stratified_samples = self._sample_stratified_from_error_pool(count)
        
        processed_samples = []
        for sample in stratified_samples:
            error_type = sample.get('error_type', 'medical')
            scribe_prompt = (
                f"Task: Rewrite this clinical note to change patient demographics, dates, "
                f"and phrasing, but KEEP the specific {error_type} error exactly as is.\n"
                f"Input Note: {sample['text']}"
            )
            ground_truth_meta = {
                "has_error": True, 
                "error_type": error_type,
                "source": "augmented_ground_truth"
            }
            
            processed_samples.append({
                "scribe_prompt": scribe_prompt,
                "meta": ground_truth_meta,
                "original_text": sample['text'],
                "mode": "augment_error"
            })
        
        return processed_samples

    def _sample_synthetic_injection(self, count: int) -> List[Dict]:
        """
        Sample for Synthetic Injection mode with error type stratification.
        
        Stratifies the injected error types across all 5 categories to ensure
        balanced representation of error types in synthetic data.
        
        Args:
            count: Number of samples to generate
            
        Returns:
            List of processed samples for synthetic injection mode
            
        Requirements: 2.7, 2.8, 2.11
        """
        if not self.clean_pool:
            return []
        
        # Sample clean notes
        samples = random.sample(self.clean_pool, min(count, len(self.clean_pool)))
        
        # Get stratified error types for injection
        stratified_error_types = self._get_stratified_error_types(len(samples))
        
        processed_samples = []
        for sample, target_error in zip(samples, stratified_error_types):
            scribe_prompt = (
                f"Task: Rewrite this healthy note to include a subtle {target_error} error.\n"
                f"Input Note: {sample['text']}"
            )
            ground_truth_meta = {
                "has_error": True, 
                "error_type": target_error,
                "source": "synthetic_injection"
            }
            
            processed_samples.append({
                "scribe_prompt": scribe_prompt,
                "meta": ground_truth_meta,
                "original_text": sample['text'],
                "mode": "inject_new_error"
            })
        
        return processed_samples

    def _sample_and_process(self, pool: List[Dict], count: int, mode: str) -> List[Dict]:
        """
        Sample data and prepare prompts for the Scribe Agent.
        
        Args:
            pool: Pool to sample from (error_pool or clean_pool)
            count: Number of samples to generate
            mode: Transformation mode (augment_error, augment_safe, make_decoy, inject_new_error)
            
        Returns:
            List of processed samples with scribe prompts and metadata
        """
        if not pool:
            return []
        
        samples = random.sample(pool, min(count, len(pool)))
        processed_samples = []

        for sample in samples:
            scribe_prompt = ""
            ground_truth_meta = {}

            if mode == "augment_error":
                error_type = sample.get('error_type', 'medical')
                scribe_prompt = (
                    f"Task: Rewrite this clinical note to change patient demographics, dates, "
                    f"and phrasing, but KEEP the specific {error_type} error exactly as is.\n"
                    f"Input Note: {sample['text']}"
                )
                ground_truth_meta = {
                    "has_error": True, 
                    "error_type": error_type,
                    "source": "augmented_ground_truth"
                }

            elif mode == "augment_safe":
                scribe_prompt = (
                    f"Task: Paraphrase this clinical note. Keep it medically accurate and error-free.\n"
                    f"Input Note: {sample['text']}"
                )
                ground_truth_meta = {
                    "has_error": False, 
                    "source": "augmented_safe"
                }

            elif mode == "make_decoy":
                scribe_prompt = (
                    f"Task: Introduce cosmetic noise (typos, formatting issues) into this note "
                    f"but DO NOT introduce any medical logic errors.\n"
                    f"Input Note: {sample['text']}"
                )
                ground_truth_meta = {
                    "has_error": False, 
                    "source": "synthetic_decoy"
                }

            elif mode == "inject_new_error":
                # Randomly pick a target error type from MEDEC categories
                target_error = random.choice(self.VALID_ERROR_TYPES)
                scribe_prompt = (
                    f"Task: Rewrite this healthy note to include a subtle {target_error} error.\n"
                    f"Input Note: {sample['text']}"
                )
                ground_truth_meta = {
                    "has_error": True, 
                    "error_type": target_error,
                    "source": "synthetic_injection"
                }

            processed_samples.append({
                "scribe_prompt": scribe_prompt,
                "meta": ground_truth_meta,
                "original_text": sample['text'],
                "mode": mode
            })
            
        return processed_samples
    
    def get_error_pool(self) -> List[Dict]:
        """Return the error pool."""
        return self.error_pool
    
    def get_clean_pool(self) -> List[Dict]:
        """Return the clean pool."""
        return self.clean_pool
    
    def get_raw_data(self) -> List[Dict]:
        """Return all raw loaded data."""
        return self.raw_data

    @classmethod
    def load_test_data(
        cls,
        data_path: str = "data_raw/MEDEC",
        subsets: Optional[List[str]] = None
    ) -> "MedicalDataProcessor":
        """
        Factory method to load only test data for evaluation.

        Args:
            data_path: Path to MEDEC data
            subsets: Subsets to load test data from

        Returns:
            MedicalDataProcessor with only test data loaded
        """
        return cls(
            data_path=data_path,
            use_huggingface=False,
            subsets=subsets or ['MS', 'UW'],
            exclude_test_files=False,
            load_test_only=True
        )

    @classmethod
    def load_training_data(
        cls,
        data_path: str = "data_raw/MEDEC",
        subsets: Optional[List[str]] = None
    ) -> "MedicalDataProcessor":
        """
        Factory method to load training data (excludes test files).

        This includes:
        - Training set files
        - Validation set files (can be used for training in RL)

        Args:
            data_path: Path to MEDEC data
            subsets: Subsets to load training data from

        Returns:
            MedicalDataProcessor with training data (no test data)
        """
        return cls(
            data_path=data_path,
            use_huggingface=False,
            subsets=subsets or ['MS', 'UW'],
            exclude_test_files=True,
            load_test_only=False
        )


# Example Usage
if __name__ == "__main__":
    print("=" * 60)
    print("Loading TRAINING data (excludes test files)")
    print("=" * 60)
    train_processor = MedicalDataProcessor.load_training_data()

    print(f"\nTraining Error pool: {len(train_processor.error_pool)}")
    print(f"Training Clean pool: {len(train_processor.clean_pool)}")

    # Generate a sample batch
    batch = train_processor.get_quadrant_batch(4)
    print(f"\nGenerated batch with {len(batch)} samples")

    print("\n" + "=" * 60)
    print("Loading TEST data (for evaluation only)")
    print("=" * 60)
    test_processor = MedicalDataProcessor.load_test_data()

    print(f"\nTest Error pool: {len(test_processor.error_pool)}")
    print(f"Test Clean pool: {len(test_processor.clean_pool)}")
