#!/usr/bin/env python3
"""
Medical Error Detection + Localization Inference Script

Tests different model versions on MEDEC test data:
- Qwen3 models (Qwen/Qwen3-4B, etc.)
- MedGemma models (google/medgemma-4b-it, google/medgemma-4b-pt)
- Fine-tuned models (from SFT/GRPO)

Uses CoT prompting for error detection + sentence-level localization.
Input: pre-numbered sentences. Output: CORRECT or sentence number.
Supports both Qwen3 thinking format and standard generation.
"""

import os
import json
import re
import sys
import argparse
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add project root to path for shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.self_play.utils import number_sentences, parse_assessor_answer

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Qwen3 special token IDs (from official documentation)
THINK_END_TOKEN_ID = 151668  # 
IM_END_TOKEN_ID = 151645  # 
