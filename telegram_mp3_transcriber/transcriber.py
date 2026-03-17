from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any

from faster_whisper import WhisperModel

from audio_utils import SAMPLE_RATE, decode_audio_mono, split_audio_by_limits

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    chunk_count: int
    used_chunking: bool


class SpeechTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        target_chunk_mb: int = 20,
        max_chunk_seconds: int = 15 * 60,
        chunk_overlap_seconds: float = 2.0,
        beam_size: int = 8,
        best_of: int = 8,
        temperature: float = 0.0,
        vad_filter: bool = False,
        condition_on_previous_text: bool = False,
    ) -> None:
        self.model_size = model_size
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.device = device
        self.compute_type = compute_type

        self.model: WhisperModel | None = None
        self._model_lock = threading.Lock()
        self._loading = False

        self.target_chunk_mb = target_chunk_mb
        self.max_chunk_seconds = max_chunk_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.beam_size = beam_size
        self.best_of = best_of
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        needles = ("cublas", "cudnn", "cuda", "onnxruntime_providers_cuda")
        return any(needle in msg for needle in needles)

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def is_loading(self) -> bool:
        return self._loading

    def status(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "requested_device": self.requested_device,
            "requested_compute_type": self.requested_compute_type,
            "active_device": self.device,
            "active_compute_type": self.compute_type,
            "model_loaded": self.is_model_loaded(),
            "loading": self.is_loading(),
        }

    def ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            self._loading = True
            try:
                self._load_model_with_fallback()
            finally:
                self._loading = False

    def _load_model_with_fallback(self) -> None:
        logger.info(
            "Loading whisper model '%s' on device=%s (compute=%s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )
        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model is ready.")
        except RuntimeError as exc:
            if self._is_cuda_runtime_error(exc) and self.device != "cpu":
                logger.warning(
                    "CUDA runtime is not available (%s). Falling back to CPU int8.",
                    exc,
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    model_size_or_path=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("Whisper model is ready on CPU fallback.")
                return
            raise

    def _reload_on_cpu(self) -> None:
        with self._model_lock:
            self.device = "cpu"
            self.compute_type = "int8"
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    @staticmethod
    def _merge_chunk_texts(chunk_texts: list[str]) -> str:
        if not chunk_texts:
            return ""
        if len(chunk_texts) == 1:
            return chunk_texts[0]

        merged_words = chunk_texts[0].split()
        for text in chunk_texts[1:]:
            words = text.split()
            if not words:
                continue

            max_overlap = min(40, len(merged_words), len(words))
            overlap = 0
            for k in range(max_overlap, 0, -1):
                left = [w.lower() for w in merged_words[-k:]]
                right = [w.lower() for w in words[:k]]
                if left == right:
                    overlap = k
                    break
            merged_words.extend(words[overlap:])
        return " ".join(merged_words).strip()

    def _decode_options(self, quality_profile: str | None) -> dict[str, Any]:
        profile = (quality_profile or "balanced").lower()
        if profile == "fast":
            return {
                "beam_size": max(2, min(self.beam_size, 4)),
                "best_of": max(2, min(self.best_of, 4)),
                "temperature": self.temperature,
                "vad_filter": True,
                "condition_on_previous_text": True,
            }
        if profile == "best":
            return {
                "beam_size": max(self.beam_size, 10),
                "best_of": max(self.best_of, 10),
                "temperature": 0.0,
                "vad_filter": False,
                "condition_on_previous_text": False,
            }
        return {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
        }

    @staticmethod
    def _initial_prompt(language: str) -> str | None:
        if language == "ru":
            return "Accurate transcription of Russian speech. Keep names, numbers, and punctuation."
        if language == "en":
            return "Accurate transcription of English speech. Keep names, numbers, and punctuation."
        return None

    def _transcribe_chunks(
        self,
        chunks: list,
        language: str,
        quality_profile: str | None,
    ) -> TranscriptionResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        chunk_texts: list[str] = []
        detected_languages: list[str] = []
        decode_options = self._decode_options(quality_profile)
        initial_prompt = self._initial_prompt(language)

        for chunk in chunks:
            segments, info = self.model.transcribe(
                chunk,
                language=None if language == "auto" else language,
                beam_size=decode_options["beam_size"],
                best_of=decode_options["best_of"],
                temperature=decode_options["temperature"],
                vad_filter=decode_options["vad_filter"],
                condition_on_previous_text=decode_options["condition_on_previous_text"],
                initial_prompt=initial_prompt,
            )
            detected_languages.append(info.language or "unknown")

            chunk_text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    chunk_text_parts.append(text)

            chunk_text = " ".join(chunk_text_parts).strip()
            if chunk_text:
                chunk_texts.append(chunk_text)

        final_text = self._merge_chunk_texts(chunk_texts)
        if not final_text:
            final_text = "No speech detected."

        lang_counter = Counter(detected_languages)
        top_language = lang_counter.most_common(1)[0][0] if lang_counter else "unknown"

        return TranscriptionResult(
            text=final_text,
            language=top_language,
            chunk_count=len(chunks),
            used_chunking=len(chunks) > 1,
        )

    def transcribe_file(
        self,
        input_file: Path,
        language: str = "auto",
        quality_profile: str | None = None,
    ) -> TranscriptionResult:
        if not input_file.exists():
            raise FileNotFoundError(f"Audio file not found: {input_file}")

        self.ensure_model_loaded()

        audio = decode_audio_mono(input_file, sampling_rate=SAMPLE_RATE)
        chunks = split_audio_by_limits(
            audio,
            file_size_bytes=input_file.stat().st_size,
            target_chunk_mb=self.target_chunk_mb,
            max_chunk_seconds=self.max_chunk_seconds,
            sampling_rate=SAMPLE_RATE,
            chunk_overlap_seconds=self.chunk_overlap_seconds,
        )

        try:
            return self._transcribe_chunks(
                chunks,
                language=language,
                quality_profile=quality_profile,
            )
        except RuntimeError as exc:
            if self._is_cuda_runtime_error(exc) and self.device != "cpu":
                logger.warning(
                    "CUDA runtime failed during transcription (%s). Falling back to CPU mode.",
                    exc,
                )
                self._reload_on_cpu()
                return self._transcribe_chunks(
                    chunks,
                    language=language,
                    quality_profile=quality_profile,
                )
            raise
