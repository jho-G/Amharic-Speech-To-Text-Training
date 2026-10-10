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


def merge_and_export_model(
    base_model_name: str,
    lora_checkpoint_dir: str,
    export_dir: str,
    device: str = "auto",
):
    """
    Merges LoRA adapter into base Whisper weights and exports for production serving.

    Args:
        base_model_name: Hugging Face model ID (e.g. 'openai/whisper-small')
        lora_checkpoint_dir: Path to trained LoRA adapter directory
        export_dir: Destination path for standalone merged model
        device: Device to use during merge ('auto', 'cuda', 'cpu')
    """
    logger.info(f"Loading base Whisper model: {base_model_name}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map=device if device != "cpu" else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    logger.info(f"Loading Whisper processor from adapter dir: {lora_checkpoint_dir}")
    processor = WhisperProcessor.from_pretrained(lora_checkpoint_dir)

    logger.info(f"Loading LoRA adapter from: {lora_checkpoint_dir}")
    lora_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)

