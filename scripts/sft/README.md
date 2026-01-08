# CoT Reasoning Trace Generation

This script generates Chain-of-Thought (CoT) reasoning traces for supervised fine-tuning (SFT) using OpenAI's GPT-4o API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Basic Usage

```bash
python scripts/sft/create_cot_reasoning_traces.py
```

### Qwen3-4B LoRA SFT (chat format)

```bash
python scripts/sft/train_qwen3_4b_sft_lora.py \
  --train-file data_processed/medec_cot/sft_cot_training_data.jsonl \
  --output-dir outputs/qwen3-4b-lora \
  --bf16
```

Notes:
- The script converts `<reasoning>` (and related tags) to Qwen `<think>` tags.
- It uses chat formatting (`system` + `user` + `assistant`) and LoRA by default.

### Configuration

Edit the `main()` function to adjust:
- `MODEL`: OpenAI model to use (default: "gpt-4o")
- `MAX_CONCURRENT`: Number of parallel API calls (default: 10)
- `BATCH_SIZE`: Number of note pairs processed concurrently (default: 5)
- `INPUT_PATH`: Path to input JSONL file
- `OUTPUT_PATH`: Path to save generated dataset

### Testing with Subset

To test with only the first 10 pairs:

```python
# In main() function, uncomment:
pairs = pairs[:10]
```

## Output Format

The script generates 4 training examples per note pair:

1. **Critic on Correct Note**: Assessment reasoning for correct clinical note
2. **Critic on Error Note**: Detection reasoning for note with error
3. **Generator for Correct Note**: Creation reasoning showing how correct note was built
4. **Generator for Error Note**: Adversarial reasoning showing how error was injected

Each example includes:
```json
{
  "note_id": "ms-train-0_critic_correct",
  "role": "critic" | "generator",
  "task": "assessment" | "generation",
  "note": "Clinical note text...",
  "reasoning": "<reasoning>...</reasoning>",
  "label": "CORRECT" | "ERROR",
  "error_type": "causalorganism" | "diagnosis" (optional),
  "error_details": {...} (optional)
}
```

## Performance

- **Parallel Processing**: Uses `asyncio` with semaphore for concurrent API calls
- **Progress Tracking**: Real-time progress bars with `tqdm`
- **Error Handling**: Graceful error logging and retry mechanisms
- **Rate Limiting**: Configurable concurrency to respect API limits

## Cost Estimation

For GPT-4o (as of 2024):
- Input: ~$2.50 per 1M tokens
- Output: ~$10.00 per 1M tokens

Estimated cost per note pair (4 examples):
- Input: ~2000 tokens × 4 = 8000 tokens
- Output: ~1500 tokens × 4 = 6000 tokens
- **Total: ~$0.08 per note pair**

For 260 pairs in your dataset: **~$21**

## Monitoring

The script provides:
- Real-time progress updates
- Success/failure logging for each generation
- Final statistics summary
- Saved output to JSONL for inspection

## Troubleshooting

### API Rate Limits
If you hit rate limits, reduce `MAX_CONCURRENT`:
```python
MAX_CONCURRENT = 5  # Lower concurrency
```

### Timeouts
Increase `max_tokens` if reasoning traces are truncated:
```python
max_tokens=4000  # Allow longer responses
```

### Out of Memory
Process in smaller batches:
```python
BATCH_SIZE = 2  # Smaller batches
```
