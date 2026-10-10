#!/usr/bin/env python3
"""
================================================================================
Production-Ready Amharic Speech-to-Text (ASR) Fine-Tuning Pipeline
================================================================================
Target Model   : openai/whisper-small (MIT License - Commercially Viable)
Target Language: Amharic (አማርኛ) - Ethiopic/Ge'ez Script
Target Dataset : leyu-amharic (Multi-dialect: Addis Ababa + Shewa)
Optimization   : Parameter-Efficient Fine-Tuning (PEFT / LoRA)
Author         : Expert ML Engineer (Speech & Audio Team)
================================================================================
"""

import os
import re
import io
import argparse
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import evaluate
from datasets import load_dataset, concatenate_datasets, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# -----------------------------------------------------------------------------
# 0. LOGGING & SYSTEM SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. TEXT NORMALIZATION (GE'EZ / AMHARIC SCRIPT)
# -----------------------------------------------------------------------------
class AmharicTextNormalizer:
    """
    Text normalizer tailored for Amharic (Ge'ez script).
    Handles traditional Ethiopic punctuation marks and whitespace normalization.
    
    In Ethiopic orthography:
      - '፡' (U+1361) is the traditional Ethiopic wordspace (ቃላት መለያ).
        Modern written Amharic frequently alternates between '፡' and standard ASCII space ' '.
        Converting '፡' to standard space ensures fair, standardized WER calculation.
      - '።' (U+1362) is the Ethiopic full stop (አራት ነጥብ).
      - '፤' (U+1364) is the Ethiopic semicolon.
      - '፣' (U+1363) is the Ethiopic comma.
    """

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        # 1. Replace traditional Ethiopic word separators with standard spaces
        text = text.replace("፡", " ")
        # 2. Normalize multiple spaces into a single space
        text = re.sub(r"\s+", " ", text)
        # 3. Strip leading and trailing whitespace
        return text.strip()


class SpeechDatasetPreprocessor:
    """
    Top-level callable preprocessor for Whisper audio features and Amharic text tokens.
    Directly decodes audio bytes using soundfile and resamples with torchaudio.
    This completely bypasses torchcodec / FFmpeg DLL dependency issues on Windows.
    """

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        target_sampling_rate: int = 16000,
        max_label_length: int = 448,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.target_sampling_rate = target_sampling_rate
        self.max_label_length = max_label_length
        self.normalizer = AmharicTextNormalizer()

