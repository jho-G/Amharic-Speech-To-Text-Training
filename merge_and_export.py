#!/usr/bin/env python3
"""
================================================================================
LoRA Weight Merge & Production Export Pipeline for Amharic Whisper
================================================================================
This script:
1. Loads the base model (openai/whisper-small, MIT License).
2. Merges the trained LoRA adapter weights directly into the base model weights
   using PEFT's `merge_and_unload()`.
3. Exports a clean, standalone Whisper model directory with zero PEFT dependency.
4. Performs sample standalone ASR inference using the merged model to verify correctness.
5. Outlines optimization paths for Faster-Whisper (CTranslate2) and TensorRT.
================================================================================
"""

import os
import argparse
import logging
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


