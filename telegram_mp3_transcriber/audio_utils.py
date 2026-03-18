from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000


def decode_audio_mono(input_path: Path, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    return decode_audio(str(input_path), sampling_rate=sampling_rate)


def split_audio_by_limits(
    audio: np.ndarray,
    file_size_bytes: int,
    target_chunk_mb: int = 20,
    max_chunk_seconds: int = 15 * 60,
    sampling_rate: int = SAMPLE_RATE,
    chunk_overlap_seconds: float = 0.0,
) -> list[tuple[np.ndarray, int]]:
    duration_seconds = len(audio) / sampling_rate if len(audio) else 0.0
    if duration_seconds <= 0:
        return [(audio, 0)]

    target_bytes = target_chunk_mb * 1024 * 1024
    if file_size_bytes <= target_bytes and duration_seconds <= max_chunk_seconds:
        return [(audio, 0)]

    size_ratio = target_bytes / max(file_size_bytes, 1)
    size_based_seconds = math.floor(duration_seconds * size_ratio * 0.95)
    segment_seconds = max(30, min(max_chunk_seconds, size_based_seconds))

    if segment_seconds >= duration_seconds:
        return [(audio, 0)]

    samples_per_chunk = max(1, int(segment_seconds * sampling_rate))
    overlap_samples = max(0, int(chunk_overlap_seconds * sampling_rate))
    overlap_samples = min(overlap_samples, max(0, samples_per_chunk - 1))
    step = max(1, samples_per_chunk - overlap_samples)

    chunks: list[tuple[np.ndarray, int]] = []
    start = 0
    while start < len(audio):
        end = min(len(audio), start + samples_per_chunk)
        chunks.append((audio[start:end], start))
        if end >= len(audio):
            break
        start += step

    if not chunks:
        return [(audio, 0)]
    return chunks
