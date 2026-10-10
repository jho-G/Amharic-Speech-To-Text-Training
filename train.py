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

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_info = batch["audio"]
        # 1. Decode raw bytes or access array
        if isinstance(audio_info, dict) and "bytes" in audio_info and audio_info["bytes"] is not None:
            array, orig_sr = sf.read(io.BytesIO(audio_info["bytes"]))
        elif isinstance(audio_info, dict) and "array" in audio_info and audio_info["array"] is not None:
            array = audio_info["array"]
            orig_sr = audio_info.get("sampling_rate", self.target_sampling_rate)
        else:
            raise ValueError(f"Unrecognized audio format: {type(audio_info)}")

        # 2. Convert multi-channel / stereo to mono
        if hasattr(array, "ndim") and array.ndim > 1:
            array = array.mean(axis=1)

        # 3. Resample to 16kHz for Whisper's log-Mel filterbanks
        if orig_sr != self.target_sampling_rate:
            waveform_t = torch.from_numpy(array.astype(np.float32))
            array = torchaudio.functional.resample(
                waveform_t, orig_freq=orig_sr, new_freq=self.target_sampling_rate
            ).numpy()

        # 4. Extract log-Mel spectrogram (Whisper expects 80 Mel bins x 3000 frames)
        batch["input_features"] = self.feature_extractor(
            array,
            sampling_rate=self.target_sampling_rate,
        ).input_features[0]

        # 5. Clean Ge'ez script text and encode to target token IDs (bounded by Whisper decoder length)
        cleaned_text = self.normalizer.normalize(batch["text"])
        batch["labels"] = self.tokenizer(
            cleaned_text,
            max_length=self.max_label_length,
            truncation=True,
        ).input_ids
        return batch


# -----------------------------------------------------------------------------
# 2. DATA COLLATOR (DYNAMIC SPEECH & SEQ2SEQ PADDING)
# -----------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads input features (audio Mel-spectrograms)
    and labels (tokenized Amharic text) to the maximum length within each batch.
    
    Crucial PyTorch Masking:
      Replaces padding token IDs in labels with -100 so that CrossEntropyLoss
      ignores padded label positions when computing backpropagation loss.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they require different padding strategies
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # 1. Pad audio input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 2. Pad label token sequences
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 so CrossEntropy loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If decoder_start_token_id is present at the beginning of labels, slice it out
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# -----------------------------------------------------------------------------
# 3. METRICS EVALUATION (WER & CER FOR GE'EZ SCRIPT)
# -----------------------------------------------------------------------------
class ComputeMetricsWrapper:
    """
    Computes Word Error Rate (WER) and Character Error Rate (CER) on decoded
    Amharic transcripts using the evaluate library.
    """

    def __init__(self, tokenizer: WhisperTokenizer):
        self.tokenizer = tokenizer
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.normalizer = AmharicTextNormalizer()

    def __call__(self, pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id so tokenizer can decode cleanly
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode tokens to Unicode text, skipping special tokens like <|startoftranscript|>
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalize Ethiopic orthography (wordspaces & punctuation)
        pred_str = [self.normalizer.normalize(text) for text in pred_str]
        label_str = [self.normalizer.normalize(text) for text in label_str]

        # Filter out empty references if any, avoiding ZeroDivisionError in jiwer
        filtered_preds, filtered_labels = [], []
        for p, l in zip(pred_str, label_str):
            if len(l.strip()) > 0:
                filtered_preds.append(p)
                filtered_labels.append(l)

        if not filtered_labels:
            logger.warning("Empty validation reference strings encountered.")
            return {"wer": 100.0, "cer": 100.0}

        # Calculate metrics as percentages (0 - 100%)
        wer = 100.0 * self.wer_metric.compute(
            predictions=filtered_preds, references=filtered_labels
        )
        cer = 100.0 * self.cer_metric.compute(
            predictions=filtered_preds, references=filtered_labels
        )

        return {"wer": round(wer, 2), "cer": round(cer, 2)}


# -----------------------------------------------------------------------------
# 4. DATASET LOADER & CONCATENATOR (DIALECT GENERALIZATION)
# -----------------------------------------------------------------------------
def load_and_prepare_leyu_dataset(
    dataset_names: List[str],
    eval_split_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Loads multiple regional dialect datasets from Hugging Face's leyu-amharic organization,
    harmonizes schemas, concatenates them for generalized acoustic modeling,
    and splits into train and validation sets.
    """
    loaded_datasets = []

    for name in dataset_names:
        # verification_mode="no_checks" avoids NonMatchingSplitsSizesError when dataset
        # maintainers upload additional samples without updating repo metadata headers.
        ds = load_dataset(name, split="train", verification_mode="no_checks")

