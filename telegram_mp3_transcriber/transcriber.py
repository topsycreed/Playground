from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from typing import Any, Callable

import numpy as np
from faster_whisper import WhisperModel

from audio_utils import SAMPLE_RATE, decode_audio_mono, split_audio_by_limits
from diarization_nemo import NemoDiarizer

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    chunk_count: int
    used_chunking: bool
    speaker_count: int = 1
    used_dialog_labels: bool = False
    diarization_backend: str = "none"


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    speaker: str = "User1"


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
        diarization_backend: str = "auto",
        nemo_num_speakers: int = 0,
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
        self.diarization_backend = (diarization_backend or "auto").lower()
        self.nemo_num_speakers = nemo_num_speakers
        self.nemo_diarizer: NemoDiarizer | None = None
        self._ensure_nemo_diarizer()

    def _ensure_nemo_diarizer(self) -> None:
        if self.diarization_backend not in {"nemo", "auto"}:
            return
        if self.nemo_diarizer is not None:
            return
        preferred_device = "cuda" if self.device in {"cuda", "auto"} else "cpu"
        self.nemo_diarizer = NemoDiarizer(preferred_device=preferred_device)

    def set_diarization_backend(self, backend: str) -> None:
        chosen = (backend or "").strip().lower()
        if chosen not in {"auto", "nemo", "heuristic"}:
            raise ValueError("Unsupported diarization backend.")
        self.diarization_backend = chosen
        if chosen == "heuristic":
            self.nemo_diarizer = None
        else:
            self._ensure_nemo_diarizer()

    @staticmethod
    def _is_cuda_runtime_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        needles = ("cublas", "cudnn", "cuda", "onnxruntime_providers_cuda")
        return any(needle in msg for needle in needles)

    def is_model_loaded(self) -> bool:
        return self.model is not None

    def is_loading(self) -> bool:
        return self._loading

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[dict[str, Any]], None] | None,
        **payload: Any,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(payload)
        except Exception:  # pylint: disable=broad-except
            logger.debug("Progress callback failed", exc_info=True)

    def status(self) -> dict[str, Any]:
        nemo_available = None
        nemo_reason = ""
        if self.diarization_backend in {"nemo", "auto"} and self.nemo_diarizer is not None:
            nemo_available, nemo_reason = self.nemo_diarizer.availability()
        return {
            "model_size": self.model_size,
            "requested_device": self.requested_device,
            "requested_compute_type": self.requested_compute_type,
            "active_device": self.device,
            "active_compute_type": self.compute_type,
            "model_loaded": self.is_model_loaded(),
            "loading": self.is_loading(),
            "diarization_backend": self.diarization_backend,
            "nemo_num_speakers": self.nemo_num_speakers,
            "nemo_available": nemo_available,
            "nemo_reason": nemo_reason,
        }

    def ensure_model_loaded(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            self._loading = True
            self._emit_progress(
                progress_callback,
                stage="model_loading",
                message="Preparing Whisper model...",
            )
            try:
                self._load_model_with_fallback()
            finally:
                self._loading = False
            self._emit_progress(
                progress_callback,
                stage="model_ready",
                message="Model is ready.",
            )

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
    def _normalize_text_for_match(text: str) -> str:
        return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).strip()

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

    @staticmethod
    def _dedupe_overlap_utterances(utterances: list[Utterance]) -> list[Utterance]:
        if not utterances:
            return []
        utterances_sorted = sorted(utterances, key=lambda u: (u.start, u.end))
        result: list[Utterance] = []
        for utt in utterances_sorted:
            if not utt.text.strip():
                continue
            if not result:
                result.append(utt)
                continue
            prev = result[-1]
            prev_norm = SpeechTranscriber._normalize_text_for_match(prev.text)
            curr_norm = SpeechTranscriber._normalize_text_for_match(utt.text)
            is_time_overlap = utt.start <= prev.end + 0.2
            is_text_duplicate = (
                curr_norm == prev_norm
                or (curr_norm and curr_norm in prev_norm)
                or (prev_norm and prev_norm in curr_norm)
            )
            if is_time_overlap and is_text_duplicate:
                continue
            result.append(utt)
        return result

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

    @staticmethod
    def _kmeans(features: np.ndarray, k: int, iterations: int = 25) -> np.ndarray:
        n = len(features)
        if n == 0:
            return np.array([], dtype=np.int32)
        if k <= 1 or n == 1:
            return np.zeros(n, dtype=np.int32)

        # Deterministic init: pick farthest points greedily.
        centers = [features[0]]
        while len(centers) < k:
            center_arr = np.vstack(centers)
            dists = np.linalg.norm(features[:, None, :] - center_arr[None, :, :], axis=2)
            min_dist = np.min(dists, axis=1)
            next_idx = int(np.argmax(min_dist))
            if any(np.allclose(features[next_idx], c) for c in centers):
                break
            centers.append(features[next_idx])

        while len(centers) < k:
            centers.append(features[len(centers) % n])
        center_arr = np.vstack(centers[:k])

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(iterations):
            dist_to_centers = np.linalg.norm(features[:, None, :] - center_arr[None, :, :], axis=2)
            new_labels = np.argmin(dist_to_centers, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_id in range(k):
                members = features[labels == cluster_id]
                if len(members) > 0:
                    center_arr[cluster_id] = np.mean(members, axis=0)
        return labels

    @staticmethod
    def _silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
        n = len(features)
        if n < 3:
            return -1.0

        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            return -1.0

        cluster_sizes = {int(lbl): int(np.sum(labels == lbl)) for lbl in unique_labels}
        if any(size <= 1 for size in cluster_sizes.values()):
            return -1.0

        dist = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
        scores = []
        for i in range(n):
            same_mask = labels == labels[i]
            same_mask[i] = False
            if not np.any(same_mask):
                continue
            a_i = float(np.mean(dist[i, same_mask]))

            b_i = float("inf")
            for lbl in unique_labels:
                if lbl == labels[i]:
                    continue
                other_mask = labels == lbl
                if not np.any(other_mask):
                    continue
                b_i = min(b_i, float(np.mean(dist[i, other_mask])))

            denom = max(a_i, b_i)
            if denom <= 1e-8:
                continue
            scores.append((b_i - a_i) / denom)

        return float(np.mean(scores)) if scores else -1.0

    def _estimate_speaker_count(self, features: np.ndarray, max_speakers: int = 4) -> tuple[int, np.ndarray]:
        n = len(features)
        if n < 3:
            return 1, np.zeros(n, dtype=np.int32)

        upper = min(max_speakers, n)
        best_k = 1
        best_labels = np.zeros(n, dtype=np.int32)
        best_score = -1.0

        for k in range(2, upper + 1):
            labels = self._kmeans(features, k)
            score = self._silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        # Conservative threshold: keep single speaker unless clustering is convincing.
        if best_score < 0.12:
            return 1, np.zeros(n, dtype=np.int32)
        return best_k, best_labels

    @staticmethod
    def _segment_embedding(audio_slice: np.ndarray) -> np.ndarray | None:
        # Very short slices are unstable for speaker features and cause label flips.
        if len(audio_slice) < int(0.6 * SAMPLE_RATE):
            return None

        clip = audio_slice.astype(np.float32)
        clip = clip - np.mean(clip)
        if len(clip) > int(6 * SAMPLE_RATE):
            clip = clip[: int(6 * SAMPLE_RATE)]

        energy = np.mean(clip * clip)
        if energy < 1e-8:
            return None

        window = np.hanning(len(clip))
        spectrum = np.abs(np.fft.rfft(clip * window))[1:]
        if spectrum.size < 32:
            return None

        bands = np.array_split(spectrum, 32)
        band_energy = np.array([np.log(np.mean(b * b) + 1e-8) for b in bands], dtype=np.float32)
        zcr = np.mean(np.abs(np.diff(np.sign(clip)))) * 0.5
        rms_log = np.log(energy + 1e-8)
        return np.concatenate([band_energy, np.array([zcr, rms_log], dtype=np.float32)])

    @staticmethod
    def _map_labels_by_first_appearance(raw_labels: list[str | int]) -> dict[str | int, str]:
        seen: list[str | int] = []
        for label in raw_labels:
            if label not in seen:
                seen.append(label)
        return {label: f"User{i + 1}" for i, label in enumerate(seen)}

    @staticmethod
    def _assign_from_diar_segments(
        utterances: list[Utterance],
        diar_segments: list[dict[str, Any]],
    ) -> tuple[list[Utterance], int]:
        if not utterances:
            return utterances, 0
        if not diar_segments:
            return utterances, 1

        assigned_raw: list[str] = []
        for utt in utterances:
            best_label = None
            best_overlap = 0.0
            for seg in diar_segments:
                overlap = min(utt.end, seg["end"]) - max(utt.start, seg["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = seg["speaker"]
            if best_label is None:
                # fallback to nearest segment by center distance
                center = (utt.start + utt.end) * 0.5
                nearest = min(
                    diar_segments,
                    key=lambda s: abs(((s["start"] + s["end"]) * 0.5) - center),
                )
                best_label = nearest["speaker"]
            assigned_raw.append(best_label)

        label_map = SpeechTranscriber._map_labels_by_first_appearance(assigned_raw)
        for utt, raw in zip(utterances, assigned_raw):
            utt.speaker = label_map[raw]

        return utterances, len(label_map)

    @staticmethod
    def _stabilize_speaker_labels(utterances: list[Utterance], labels: list[int]) -> list[int]:
        if len(labels) <= 2:
            return labels

        stabilized = labels[:]
        changed = True
        passes = 0
        while changed and passes < 4:
            changed = False
            passes += 1
            for i in range(1, len(stabilized) - 1):
                left = stabilized[i - 1]
                cur = stabilized[i]
                right = stabilized[i + 1]
                if cur == left or cur == right:
                    continue

                dur = utterances[i].end - utterances[i].start
                words = len(utterances[i].text.split())
                if left == right and (dur <= 2.2 or words <= 7):
                    stabilized[i] = left
                    changed = True

        # Remove isolated singleton islands if neighbors agree.
        for i in range(1, len(stabilized) - 1):
            if stabilized[i - 1] == stabilized[i + 1] != stabilized[i]:
                stabilized[i] = stabilized[i - 1]

        return stabilized

    def _assign_speakers(
        self,
        utterances: list[Utterance],
        audio: np.ndarray,
        input_file: Path,
        nemo_num_speakers: int | None = None,
    ) -> tuple[list[Utterance], int, str]:
        if len(utterances) <= 1:
            return utterances, 1, "heuristic"

        attempted_nemo = False
        if self.diarization_backend in {"nemo", "auto"} and self.nemo_diarizer is not None:
            attempted_nemo = True
            requested_num_speakers = (
                nemo_num_speakers
                if nemo_num_speakers is not None and nemo_num_speakers > 0
                else (self.nemo_num_speakers if self.nemo_num_speakers > 0 else None)
            )
            try:
                diar_segments = self.nemo_diarizer.diarize(
                    input_file,
                    num_speakers=requested_num_speakers,
                )
                nemo_utterances, nemo_speaker_count = self._assign_from_diar_segments(
                    utterances,
                    diar_segments,
                )
                if nemo_speaker_count >= 1:
                    return nemo_utterances, nemo_speaker_count, "nemo"
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("NeMo diarization unavailable, falling back to heuristic: %s", exc)
                if self.diarization_backend == "nemo":
                    # Explicit nemo mode still degrades gracefully in this build.
                    pass

        valid_indices: list[int] = []
        feature_rows: list[np.ndarray] = []

        for idx, utt in enumerate(utterances):
            start_idx = max(0, int(utt.start * SAMPLE_RATE))
            end_idx = min(len(audio), int(utt.end * SAMPLE_RATE))
            if end_idx <= start_idx:
                continue
            emb = self._segment_embedding(audio[start_idx:end_idx])
            if emb is None:
                continue
            valid_indices.append(idx)
            feature_rows.append(emb)

        if len(feature_rows) < 2:
            if attempted_nemo:
                return utterances, 1, "heuristic (nemo-fallback)"
            return utterances, 1, "heuristic"

        features = np.vstack(feature_rows).astype(np.float32)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-6
        features = (features - mean) / std

        speaker_count, cluster_labels = self._estimate_speaker_count(features, max_speakers=4)
        if speaker_count <= 1:
            if attempted_nemo:
                return utterances, 1, "heuristic (nemo-fallback)"
            return utterances, 1, "heuristic"

        assigned = [0] * len(utterances)
        known_positions = set(valid_indices)
        for idx, cluster in zip(valid_indices, cluster_labels):
            assigned[idx] = int(cluster)

        # Fill unknown/short segments by nearest known neighbor.
        for i in range(len(assigned)):
            if i in known_positions:
                continue
            prev_i = i - 1
            next_i = i + 1
            while prev_i >= 0 and prev_i not in known_positions:
                prev_i -= 1
            while next_i < len(assigned) and next_i not in known_positions:
                next_i += 1
            if prev_i >= 0:
                assigned[i] = assigned[prev_i]
            elif next_i < len(assigned):
                assigned[i] = assigned[next_i]
            else:
                assigned[i] = 0

        assigned = self._stabilize_speaker_labels(utterances, assigned)

        # Stable speaker names by first appearance.
        first_seen = []
        for label in assigned:
            if label not in first_seen:
                first_seen.append(label)
        if len(first_seen) <= 1:
            return utterances, 1, "heuristic"
        label_map = {label: f"User{idx + 1}" for idx, label in enumerate(first_seen)}

        for i, utt in enumerate(utterances):
            utt.speaker = label_map.get(assigned[i], "User1")
        if attempted_nemo:
            return utterances, len(first_seen), "heuristic (nemo-fallback)"
        return utterances, len(first_seen), "heuristic"

    @staticmethod
    def _merge_dialogue_turns(utterances: list[Utterance], max_gap: float = 2.4) -> list[Utterance]:
        if not utterances:
            return []
        merged = [utterances[0]]
        for utt in utterances[1:]:
            prev = merged[-1]
            if utt.speaker == prev.speaker and utt.start - prev.end <= max_gap:
                prev.text = f"{prev.text} {utt.text}".strip()
                prev.end = max(prev.end, utt.end)
            else:
                merged.append(utt)
        return merged

    def _render_dialogue(self, utterances: list[Utterance]) -> str:
        if not utterances:
            return "No speech detected."
        lines = [f"{utt.speaker}: {utt.text}" for utt in utterances if utt.text.strip()]
        return "\n".join(lines).strip() or "No speech detected."

    def _transcribe_chunks(
        self,
        chunk_items: list[tuple[np.ndarray, int]],
        audio: np.ndarray,
        input_file: Path,
        language: str,
        quality_profile: str | None,
        output_format: str,
        nemo_num_speakers: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TranscriptionResult:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        detected_languages: list[str] = []
        decode_options = self._decode_options(quality_profile)
        initial_prompt = self._initial_prompt(language)
        utterances: list[Utterance] = []
        total_chunks = len(chunk_items)
        self._emit_progress(
            progress_callback,
            stage="transcribing",
            done_chunks=0,
            total_chunks=total_chunks,
            message="Transcription started...",
        )

        for idx, (chunk, chunk_start_sample) in enumerate(chunk_items, start=1):
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

            chunk_start_sec = chunk_start_sample / SAMPLE_RATE
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                start = max(0.0, chunk_start_sec + float(segment.start))
                end = max(start, chunk_start_sec + float(segment.end))
                utterances.append(Utterance(start=start, end=end, text=text))
            self._emit_progress(
                progress_callback,
                stage="transcribing",
                done_chunks=idx,
                total_chunks=total_chunks,
                message=f"Transcribing chunk {idx}/{total_chunks}...",
            )

        utterances = self._dedupe_overlap_utterances(utterances)

        speaker_count = 1
        used_dialog_labels = False
        diarization_backend = "none"
        if output_format == "dialog":
            self._emit_progress(
                progress_callback,
                stage="diarization",
                done_chunks=total_chunks,
                total_chunks=total_chunks,
                message="Detecting speakers...",
            )
            utterances, speaker_count, diarization_backend = self._assign_speakers(
                utterances,
                audio=audio,
                input_file=input_file,
                nemo_num_speakers=nemo_num_speakers,
            )
            utterances = self._merge_dialogue_turns(utterances)
            if speaker_count > 1:
                final_text = self._render_dialogue(utterances)
                used_dialog_labels = True
            else:
                flat_texts = [utt.text for utt in utterances]
                final_text = self._merge_chunk_texts(flat_texts)
                if not final_text:
                    final_text = "No speech detected."
        else:
            flat_texts = [utt.text for utt in utterances]
            final_text = self._merge_chunk_texts(flat_texts)
            if not final_text:
                final_text = "No speech detected."

        lang_counter = Counter(detected_languages)
        top_language = lang_counter.most_common(1)[0][0] if lang_counter else "unknown"
        self._emit_progress(
            progress_callback,
            stage="finalizing",
            done_chunks=total_chunks,
            total_chunks=total_chunks,
            message="Finalizing transcript...",
        )

        return TranscriptionResult(
            text=final_text,
            language=top_language,
            chunk_count=len(chunk_items),
            used_chunking=len(chunk_items) > 1,
            speaker_count=speaker_count,
            used_dialog_labels=used_dialog_labels,
            diarization_backend=diarization_backend,
        )

    def transcribe_file(
        self,
        input_file: Path,
        language: str = "auto",
        quality_profile: str | None = None,
        output_format: str = "text",
        nemo_num_speakers: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TranscriptionResult:
        if not input_file.exists():
            raise FileNotFoundError(f"Audio file not found: {input_file}")

        self.ensure_model_loaded(progress_callback=progress_callback)

        audio = decode_audio_mono(input_file, sampling_rate=SAMPLE_RATE)
        chunk_items = split_audio_by_limits(
            audio,
            file_size_bytes=input_file.stat().st_size,
            target_chunk_mb=self.target_chunk_mb,
            max_chunk_seconds=self.max_chunk_seconds,
            sampling_rate=SAMPLE_RATE,
            chunk_overlap_seconds=self.chunk_overlap_seconds,
        )

        safe_format = output_format if output_format in {"text", "dialog"} else "text"

        try:
            return self._transcribe_chunks(
                chunk_items=chunk_items,
                audio=audio,
                input_file=input_file,
                language=language,
                quality_profile=quality_profile,
                output_format=safe_format,
                nemo_num_speakers=nemo_num_speakers,
                progress_callback=progress_callback,
            )
        except RuntimeError as exc:
            if self._is_cuda_runtime_error(exc) and self.device != "cpu":
                logger.warning(
                    "CUDA runtime failed during transcription (%s). Falling back to CPU mode.",
                    exc,
                )
                self._reload_on_cpu()
                return self._transcribe_chunks(
                    chunk_items=chunk_items,
                    audio=audio,
                    input_file=input_file,
                    language=language,
                    quality_profile=quality_profile,
                    output_format=safe_format,
                    nemo_num_speakers=nemo_num_speakers,
                    progress_callback=progress_callback,
                )
            raise
