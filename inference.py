#!/usr/bin/env python3
"""
================================================================================
Production Amharic Whisper ASR Inference Script
================================================================================
Runs fast, accurate speech-to-text inference on Amharic audio files.
"""

import sys
import argparse
import torch
import torchaudio
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration


def transcribe_audio(
    model_path: str,
    audio_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """
    Transcribes Amharic speech to Ge'ez text using the fine-tuned model.
    """
    print(f"Loading Amharic Whisper model from: {model_path} on {device}...")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device=0 if "cuda" in device else -1,
        generate_kwargs={
            "language": "amharic",
            "task": "transcribe",
            "forced_decoder_ids": None,
        },
    )

    print(f"Transcribing audio file: {audio_path}...")
    result = pipe(audio_path, return_timestamps=False)
    transcript = result["text"].strip()
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Amharic ASR Inference CLI")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./whisper-small-amharic-production",
        help="Path to merged production model directory.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to Amharic speech audio file (.wav, .mp3, .flac).",
    )
    args = parser.parse_args()

    transcription = transcribe_audio(args.model_path, args.audio_path)
    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT (የተተረጎመው ጽሑፍ):")
    print("=" * 60)
    print(transcription)
    print("=" * 60)


if __name__ == "__main__":
    main()
