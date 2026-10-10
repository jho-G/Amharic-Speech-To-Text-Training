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

