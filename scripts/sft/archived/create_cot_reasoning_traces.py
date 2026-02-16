"""
Generate Chain-of-Thought reasoning traces from existing correct/incorrect note pairs.
Uses OpenAI API (latest version with parallel processing) for distillation.

This creates a complete training dataset with:
1. Critic CoT (how to assess notes) - for both correct and error cases
2. Generator CoT (how to create notes) - for both correct and error injection
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import argparse
import logging
from dataclasses import dataclass, asdict
import time
import random

from openai import AsyncOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Configure logging - reduce verbosity during async operations
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to reduce overhead
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a separate logger for progress updates
progress_logger = logging.getLogger('progress')
progress_logger.setLevel(logging.INFO)


def list_medec_data_files(data_dir: str = 'data_raw/MEDEC'):
    """List all available MEDEC data files."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    print("\n" + "="*60)
    print(f"MEDEC DATA FILES IN: {data_dir}")
    print("="*60)
    
    # List all subdirectories and files
    for item in sorted(data_path.rglob('*.jsonl')):
        rel_path = item.relative_to(data_path)
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  {rel_path} ({size_mb:.2f} MB)")
    
    print("="*60 + "\n")


@dataclass
class CoTExample:
    """Data class for a single CoT training example."""
    note_id: str
    role: str  # 'critic' or 'generator'
    task: str  # 'assessment' or 'generation'
    note: str
    reasoning: str
    label: str  # 'CORRECT' or 'ERROR'
    error_type: Optional[str] = None
    error_details: Optional[Dict] = None
    correct_note: Optional[str] = None  # For generator error injection


class PromptLoader:
    """Load and manage prompt templates."""
    
    def __init__(self, prompts_dir: Path = Path('prompts')):
        self.prompts_dir = prompts_dir
        self.prompts = self._load_all_prompts()
    
    def _load_all_prompts(self) -> Dict[str, str]:
        """Load all prompt templates from the prompts directory."""
        logger.info(f"Loading prompts from {self.prompts_dir}")
        prompts = {
            'critic': self._load_prompt('generate_cot_from_pairs.md'),
            'generator_correct': self._load_prompt('01_correct_note_generator.md'),
            'generator_error': self._load_prompt('02_error_injection_with_reasoning.md'),
        }
        logger.info(f"✓ Loaded {len(prompts)} prompt templates")
        return prompts
    
    def _load_prompt(self, filename: str) -> str:
        """Load a single prompt template."""
        filepath = self.prompts_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")
        return filepath.read_text()


class CoTGenerator:
    """Generate CoT reasoning traces using OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_concurrent: int = 10,  # Increased default
        temperature: float = 0.7,
        max_tokens: int = 3000,
        max_retries: int = 5
    ):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Check your .env file.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        progress_logger.info(f"✓ Initialized CoTGenerator with model: {model}")
        progress_logger.info(f"  Max concurrent requests: {max_concurrent}")
    
    async def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        note_id: str
    ) -> str:
        """
        Generate a single CoT reasoning trace with retry logic.
        
        Args:
            system_prompt: The system message (prompt template)
            user_prompt: The user message (specific case)
            note_id: Identifier for logging
        
        Returns:
            Generated reasoning trace as string
        """
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                    reasoning = response.choices[0].message.content
                    return reasoning
                    
                except RateLimitError as e:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: 2^attempt * (1 + random jitter)
                        wait_time = (2 ** attempt) * (1 + random.random() * 0.1)
                        logger.warning(f"Rate limit hit for {note_id}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"✗ Max retries exceeded for {note_id}: {e}")
                        raise
                        
                except APIError as e:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"API error for {note_id}, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"✗ Max retries exceeded for {note_id}: {e}")
                        raise
                        
                except Exception as e:
                    logger.error(f"✗ Error generating reasoning for {note_id}: {e}")
                    raise
    
    async def generate_batch(
        self,
        prompts: List[tuple],
    ) -> List[str]:
        tasks = [
            self.generate_reasoning(system_prompt, user_prompt, note_id)
            for system_prompt, user_prompt, note_id in prompts
        ]
        results = await asyncio.gather(*tasks)
        return results


class PromptBuilder:
    """Build specific prompts for each CoT generation task."""
    
    @staticmethod
    def build_critic_correct_prompt(pair: Dict, template: str) -> str:
        return f"""**Constraints**:
- Limit reasoning to ≤8 bullet points, ≤200 words total.
- Focus on key clinical checks only.

**Input**:
- Clinical Note: {pair['correct_note']}
- Ground Truth: CORRECT

Generate the reasoning trace following the CORRECT note format from the template above.
"""
    
    @staticmethod
    def build_critic_error_prompt(pair: Dict, template: str) -> str:
        return f"""**Constraints**:
- Limit reasoning to ≤8 bullet points, ≤200 words total.
- Highlight only the decisive findings tied to the error.

**Input**:
- Clinical Note: {pair['incorrect_note']}
- Ground Truth: ERROR
- Error Type: {pair['error_type']}
- Error Sentence: {pair['error_sentence']}
- Corrected Sentence: {pair['corrected_sentence']}

Generate the reasoning trace following the ERROR note format from the template above.
"""
    
    @staticmethod
    def build_generator_correct_prompt(pair: Dict, template: str) -> str:
        return f"""**Constraints**:
- Limit reasoning to ≤9 bullet points, ≤250 words total.
- Summarize the drafting logic without repetitive phrasing.

**Input**:
- Clinical Note: {pair['correct_note']}

Reverse-engineer the reasoning that led to creating this correct note.
Generate the <generation_reasoning> trace showing the thought process.
"""
    
    @staticmethod
    def build_generator_error_prompt(pair: Dict, template: str) -> str:
        return f"""**Constraints**:
- Limit reasoning to ≤10 bullet points, ≤300 words total.
- Explain only the modeling decisions that made the error compelling for a note-writing model.

**Input**:
- Correct Note: {pair['correct_note']}
- Incorrect Note: {pair['incorrect_note']}
- Error Metadata: {{
    "error_type": "{pair['error_type']}",
    "error_sentence": "{pair['error_sentence']}",
    "corrected_sentence": "{pair['corrected_sentence']}"
  }}

Reconstruct the model-like reasoning that would intentionally adopt this error as a plausible alternative.
Focus on why the deviation looks convincing to an automated note generator.
Generate the <error_injection_reasoning> trace showing this thought process.
"""


class DatasetProcessor:
    """Process note pairs and generate complete CoT dataset."""
    
    def __init__(
        self,
        prompt_loader: PromptLoader,
        cot_generator: CoTGenerator,
        batch_size: int = 5  # Process N pairs concurrently
    ):
        self.prompt_loader = prompt_loader
        self.cot_generator = cot_generator
        self.prompt_builder = PromptBuilder()
        self.batch_size = batch_size
    
    def load_note_pairs(self, jsonl_path: str, limit: Optional[int] = None) -> List[Dict]:
        """Load note pairs from JSONL file."""
        pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                pairs.append(json.loads(line))
                if limit and len(pairs) >= limit:
                    break
        
        progress_logger.info(f"Loaded {len(pairs)} note pairs from {jsonl_path}")
        return pairs
    
    def create_generation_tasks(self, pair: Dict) -> List[tuple]:
        """
        Create all 4 generation tasks for a single note pair.
        
        Returns:
            List of (system_prompt, user_prompt, note_id) tuples
        """
        tasks = []
        prompts = self.prompt_loader.prompts
        
        # Task 1: Critic assessing CORRECT note
        tasks.append((
            prompts['critic'],
            self.prompt_builder.build_critic_correct_prompt(pair, prompts['critic']),
            f"{pair['note_id']}_critic_correct"
        ))
        
        # Task 2: Critic assessing ERROR note
        tasks.append((
            prompts['critic'],
            self.prompt_builder.build_critic_error_prompt(pair, prompts['critic']),
            f"{pair['note_id']}_critic_error"
        ))
        
        # Task 3: Generator creating CORRECT note
        tasks.append((
            prompts['generator_correct'],
            self.prompt_builder.build_generator_correct_prompt(pair, prompts['generator_correct']),
            f"{pair['note_id']}_generator_correct"
        ))
        
        # Task 4: Generator injecting ERROR
        tasks.append((
            prompts['generator_error'],
            self.prompt_builder.build_generator_error_prompt(pair, prompts['generator_error']),
            f"{pair['note_id']}_generator_error"
        ))
        
        return tasks
    
    def package_results(
        self,
        pair: Dict,
        reasonings: List[str]
    ) -> List[CoTExample]:
        """
        Package generated reasonings into structured training examples.
        
        Args:
            pair: Original note pair
            reasonings: List of 4 generated reasoning traces (in order)
        
        Returns:
            List of 4 CoTExample objects
        """
        critic_correct_reasoning, critic_error_reasoning, gen_correct_reasoning, gen_error_reasoning = reasonings
        
        examples = [
            # 1. Critic on correct note
            CoTExample(
                note_id=f"{pair['note_id']}_critic_correct",
                role='critic',
                task='assessment',
                note=pair['correct_note'],
                reasoning=critic_correct_reasoning,
                label='CORRECT'
            ),
            
            # 2. Critic on error note
            CoTExample(
                note_id=f"{pair['note_id']}_critic_error",
                role='critic',
                task='assessment',
                note=pair['incorrect_note'],
                reasoning=critic_error_reasoning,
                label='ERROR',
                error_type=pair['error_type'],
                error_details={
                    'error_sentence': pair['error_sentence'],
                    'corrected_sentence': pair['corrected_sentence']
                }
            ),
            
            # 3. Generator for correct note
            CoTExample(
                note_id=f"{pair['note_id']}_generator_correct",
                role='generator',
                task='generation',
                note=pair['correct_note'],
                reasoning=gen_correct_reasoning,
                label='CORRECT'
            ),
            
            # 4. Generator for error injection
            CoTExample(
                note_id=f"{pair['note_id']}_generator_error",
                role='generator',
                task='generation',
                note=pair['incorrect_note'],
                reasoning=gen_error_reasoning,
                label='ERROR',
                error_type=pair['error_type'],
                error_details={
                    'error_sentence': pair['error_sentence'],
                    'corrected_sentence': pair['corrected_sentence']
                },
                correct_note=pair['correct_note']
            )
        ]
        
        return examples
    
    async def process_pair(self, pair: Dict, silent: bool = False) -> List[CoTExample]:
        """Process a single note pair and generate all 4 CoT examples."""
        if not silent:
            logger.info(f"Processing pair: {pair['note_id']}")
        
        # Create generation tasks
        tasks = self.create_generation_tasks(pair)
        
        # Generate all reasonings in parallel with timing
        start_time = time.time()
        reasonings = await self.cot_generator.generate_batch(tasks)
        elapsed_time = time.time() - start_time
        
        # Package into training examples
        examples = self.package_results(pair, reasonings)
        
        return examples
    
    async def process_batch(self, pairs: List[Dict]) -> List[CoTExample]:
        """Process a batch of pairs concurrently."""
        tasks = [self.process_pair(pair, silent=True) for pair in pairs]
        results = await asyncio.gather(*tasks)
        # Flatten list of lists
        all_examples = []
        for examples in results:
            all_examples.extend(examples)
        return all_examples
    
    async def process_all_pairs(
        self,
        pairs: List[Dict],
        output_path: Optional[str] = None,
        save_every_batch: bool = False
    ) -> List[CoTExample]:
        """
        Process all note pairs in batches for better throughput.
        
        Args:
            pairs: List of note pairs
            output_path: Optional JSONL file to receive incremental saves
            save_every_batch: Persist each processed batch when True
        """
        all_examples = []
        total_start_time = time.time()
        
        # Split pairs into batches
        num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        
        # Use tqdm progress bar for overall progress
        with tqdm(total=len(pairs), desc="Processing pairs", unit="pair") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(pairs))
                batch = pairs[start_idx:end_idx]
                
                batch_start_time = time.time()
                
                # Process batch concurrently
                batch_examples = await self.process_batch(batch)
                
                batch_elapsed = time.time() - batch_start_time
                all_examples.extend(batch_examples)
                if save_every_batch and output_path:
                    self.save_dataset(batch_examples, output_path, append=True)
                
                # Calculate ETA
                pairs_processed = end_idx
                avg_time_per_pair = (time.time() - total_start_time) / pairs_processed
                remaining_pairs = len(pairs) - pairs_processed
                eta_seconds = avg_time_per_pair * remaining_pairs
                eta_minutes = eta_seconds / 60
                eta_hours = eta_minutes / 60
                
                # Format ETA based on magnitude
                if eta_hours >= 1:
                    eta_str = f"{eta_hours:.1f}h"
                elif eta_minutes >= 1:
                    eta_str = f"{eta_minutes:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"
                
                # Update progress bar
                pbar.set_postfix({
                    'batch_time': f'{batch_elapsed:.1f}s',
                    'avg/pair': f'{avg_time_per_pair:.1f}s',
                    'eta': eta_str
                })
                pbar.update(len(batch))
                
                # Log progress periodically
                if batch_idx % 10 == 0 or end_idx == len(pairs):
                    progress_logger.info(f"Processed {pairs_processed}/{len(pairs)} pairs | "
                                       f"Examples: {len(all_examples)} | "
                                       f"ETA: {eta_str}")
        
        return all_examples
    
    def save_dataset(self, examples: List[CoTExample], output_path: str, append: bool = False):
        """Save generated dataset to JSONL file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mode = 'a' if append else 'w'
        with open(output_path, mode) as f:
            for example in examples:
                # Convert dataclass to dict, filtering out None values
                example_dict = {k: v for k, v in asdict(example).items() if v is not None}
                f.write(json.dumps(example_dict) + '\n')
        
        action = "Appended" if append else "Saved"
        progress_logger.info(f"✓ {action} {len(examples)} examples to {output_path}")
    
    def print_statistics(self, examples: List[CoTExample]):
        """Print dataset statistics."""
        critic_examples = [ex for ex in examples if ex.role == 'critic']
        generator_examples = [ex for ex in examples if ex.role == 'generator']
        correct_examples = [ex for ex in examples if ex.label == 'CORRECT']
        error_examples = [ex for ex in examples if ex.label == 'ERROR']
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total examples generated: {len(examples)}")
        print(f"\nBy Role:")
        print(f"  - Critic (assessment):  {len(critic_examples)}")
        print(f"  - Generator (creation): {len(generator_examples)}")
        print(f"\nBy Label:")
        print(f"  - CORRECT notes: {len(correct_examples)}")
        print(f"  - ERROR notes:   {len(error_examples)}")
        print(f"\nBreakdown:")
        print(f"  - Critic on correct notes:     {len([ex for ex in critic_examples if ex.label == 'CORRECT'])}")
        print(f"  - Critic on error notes:       {len([ex for ex in critic_examples if ex.label == 'ERROR'])}")
        print(f"  - Generator for correct notes: {len([ex for ex in generator_examples if ex.label == 'CORRECT'])}")
        print(f"  - Generator for error notes:   {len([ex for ex in generator_examples if ex.label == 'ERROR'])}")
        print("="*60 + "\n")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate CoT reasoning traces from note pairs.")
    parser.add_argument("--input", "-i", type=str, 
                        default='data_processed/medec_paired/train_val_split/sft_train.jsonl',
                        help="Path to input JSONL file containing note pairs")
    parser.add_argument("--output", "-o", type=str, 
                        default='data_processed/medec_cot/sft_cot_training_data.jsonl',
                        help="Path to save the generated CoT dataset")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o",
                        help="OpenAI model to use for generation")
    parser.add_argument("--concurrent", "-c", type=int, default=10,
                        help="Maximum number of concurrent API requests")
    parser.add_argument("--batch_size", "-b", type=int, default=5,
                        help="Number of pairs to process concurrently")
    parser.add_argument("--num_pairs", "-n", type=int, default=10,
                        help="Number of pairs to process (default: 10, set 0 for all)")
    
    args = parser.parse_args()

    # Configuration 
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    MODEL = args.model
    MAX_CONCURRENT = args.concurrent
    BATCH_SIZE = args.batch_size
    # If 0 is passed, process all pairs
    NUM_PAIRS = args.num_pairs if args.num_pairs > 0 else None
    
    print("\n" + "="*60)
    print("CoT REASONING TRACE GENERATION - SFT SPLIT")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Batch size (pairs): {BATCH_SIZE}")
    print(f"Input: {INPUT_PATH}")
    print(f"Processing: {'ALL pairs' if NUM_PAIRS is None else f'{NUM_PAIRS} pairs'}")
    print(f"Output: {OUTPUT_PATH}")
    print("="*60 + "\n")
    
    # Initialize components
    progress_logger.info("Initializing CoT generation pipeline...")
    prompt_loader = PromptLoader()
    cot_generator = CoTGenerator(
        model=MODEL,
        max_concurrent=MAX_CONCURRENT,
        temperature=0.7,
        max_tokens=3000,
        max_retries=5
    )
    processor = DatasetProcessor(prompt_loader, cot_generator, batch_size=BATCH_SIZE)
    
    # Load note pairs (limit to NUM_PAIRS for testing)
    pairs = processor.load_note_pairs(INPUT_PATH, limit=NUM_PAIRS)
    
    # Process all pairs
    progress_logger.info(f"\nStarting CoT generation for {len(pairs)} note pairs...\n")
    
    examples = await processor.process_all_pairs(
        pairs,
        output_path=OUTPUT_PATH,
        save_every_batch=True
    )
    
    # Save dataset
    processor.save_dataset(examples, OUTPUT_PATH)
    
    # Print statistics
    processor.print_statistics(examples)
    
    print("\n" + "="*60)
    print("✓ CoT GENERATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Generated {len(examples)} training examples from {len(pairs)} note pairs")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("="*60 + "\n")


if __name__ == '__main__':
    asyncio.run(main())
