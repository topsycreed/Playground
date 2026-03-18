from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tempfile
from typing import Any, Callable
import wave

import numpy as np
import requests
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

NEMO_CONFIG_URL = (
    "https://raw.githubusercontent.com/NVIDIA/NeMo/main/"
    "examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml"
)


class NemoDiarizer:
    def __init__(
        self,
        preferred_device: str = "cuda",
        config_cache_dir: Path | None = None,
        msdd_model_path: str = "diar_msdd_telephonic",
        speaker_embedding_model: str = "titanet_large",
        vad_model: str = "vad_multilingual_marblenet",
    ) -> None:
        self.preferred_device = preferred_device
        self.msdd_model_path = msdd_model_path
        self.speaker_embedding_model = speaker_embedding_model
        self.vad_model = vad_model
        self._available: bool | None = None
        self._reason: str = ""
        self._diarizer_ctor: Callable[..., Any] | None = None
        self._diarizer_name: str = ""

        root = config_cache_dir or (Path.cwd() / ".cache_nemo")
        self.config_cache_dir = root
        try:
            self.config_cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.config_cache_dir = Path(tempfile.gettempdir()) / "telegram_mp3_transcriber_nemo"
            self.config_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_cache_dir / "diar_infer_meeting.yaml"

    @staticmethod
    def _nemo_num_workers() -> int:
        raw = os.getenv("NEMO_NUM_WORKERS", "").strip()
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        # Windows + NeMo often fails with multiprocessing pickling issues.
        return 0 if os.name == "nt" else 1

    def availability(self) -> tuple[bool, str]:
        if self._available is not None:
            return self._available, self._reason

        if sys.version_info >= (3, 13):
            self._available = False
            self._reason = (
                "NeMo diarization is typically unavailable on Python 3.13+ in this setup. "
                "Use Python 3.11/3.12 environment for NeMo."
            )
            return self._available, self._reason

        # On Windows, ClusteringDiarizer is generally more stable than NeuralDiarizer.
        if os.name == "nt":
            try:
                from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore

                self._diarizer_ctor = ClusteringDiarizer
                self._diarizer_name = "ClusteringDiarizer"
                self._available = True
                self._reason = ""
                return self._available, self._reason
            except Exception:
                pass

            try:
                from nemo.collections.asr.models import NeuralDiarizer  # type: ignore

                self._diarizer_ctor = NeuralDiarizer
                self._diarizer_name = "NeuralDiarizer"
                self._available = True
                self._reason = ""
                return self._available, self._reason
            except Exception as exc:
                self._available = False
                self._reason = f"NeMo diarization dependencies are missing: {exc}"
                return self._available, self._reason

        try:
            from nemo.collections.asr.models import NeuralDiarizer  # type: ignore

            self._diarizer_ctor = NeuralDiarizer
            self._diarizer_name = "NeuralDiarizer"
            self._available = True
            self._reason = ""
            return self._available, self._reason
        except Exception:
            pass

        try:
            from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore

            self._diarizer_ctor = ClusteringDiarizer
            self._diarizer_name = "ClusteringDiarizer"
            self._available = True
            self._reason = ""
            return self._available, self._reason
        except Exception as exc:
            self._available = False
            self._reason = f"NeMo diarization dependencies are missing: {exc}"
            return self._available, self._reason

    def _ensure_config_file(self) -> Path:
        if self.config_path.exists():
            return self.config_path

        logger.info("Downloading NeMo diarization config: %s", NEMO_CONFIG_URL)
        response = requests.get(NEMO_CONFIG_URL, timeout=60)
        response.raise_for_status()
        self.config_path.write_text(response.text, encoding="utf-8")
        return self.config_path

    @staticmethod
    def _resolve_device(preferred_device: str) -> str:
        import torch

        if preferred_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_cfg(
        self,
        manifest_path: Path,
        out_dir: Path,
        num_speakers: int | None,
    ) -> Any:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(self._ensure_config_file()))
        cfg.device = self._resolve_device(self.preferred_device)
        num_workers = self._nemo_num_workers()
        cfg.num_workers = num_workers

        cfg.diarizer.manifest_filepath = str(manifest_path)
        cfg.diarizer.out_dir = str(out_dir)
        cfg.diarizer.oracle_vad = False
        if hasattr(cfg.diarizer, "num_workers"):
            cfg.diarizer.num_workers = num_workers

        if hasattr(cfg.diarizer, "speaker_embeddings"):
            cfg.diarizer.speaker_embeddings.model_path = self.speaker_embedding_model
        if hasattr(cfg.diarizer, "msdd_model"):
            cfg.diarizer.msdd_model.model_path = self.msdd_model_path
        if hasattr(cfg.diarizer, "vad"):
            cfg.diarizer.vad.model_path = self.vad_model
            if hasattr(cfg.diarizer.vad, "num_workers"):
                cfg.diarizer.vad.num_workers = num_workers

        if (
            hasattr(cfg.diarizer, "clustering")
            and hasattr(cfg.diarizer.clustering, "parameters")
            and hasattr(cfg.diarizer.clustering.parameters, "oracle_num_speakers")
        ):
            cfg.diarizer.clustering.parameters.oracle_num_speakers = bool(
                num_speakers is not None and num_speakers > 0
            )

        return cfg

    @staticmethod
    def _write_manifest(
        manifest_path: Path,
        audio_path: Path,
        num_speakers: int | None,
    ) -> None:
        meta = {
            "audio_filepath": str(audio_path),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers if num_speakers and num_speakers > 0 else None,
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    @staticmethod
    def _parse_rttm(rttm_path: Path) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        if not rttm_path.exists():
            return segments

        for line in rttm_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append(
                {
                    "start": start,
                    "end": start + duration,
                    "speaker": speaker,
                }
            )
        segments.sort(key=lambda x: (x["start"], x["end"]))
        return segments

    @staticmethod
    def _prepare_audio_for_nemo(source_path: Path, out_dir: Path) -> Path:
        """
        Convert any input audio to local mono 16k WAV to reduce external codec/tool dependencies
        (notably ffprobe on locked-down Windows setups).
        """
        target_wav = out_dir / "nemo_input.wav"
        samples = decode_audio(str(source_path), sampling_rate=16000)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=0)
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(target_wav), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm16.tobytes())
        return target_wav

    def diarize(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
    ) -> list[dict[str, Any]]:
        available, reason = self.availability()
        if not available or self._diarizer_ctor is None:
            raise RuntimeError(reason or "NeMo diarization is unavailable.")

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "nemo_out"
            manifest_path = tmp_path / "manifest.json"
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                prepared_audio = self._prepare_audio_for_nemo(audio_path, tmp_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Audio pre-conversion for NeMo failed (%s). Using original file.", exc)
                prepared_audio = audio_path

            self._write_manifest(manifest_path, prepared_audio, num_speakers)
            cfg = self._build_cfg(manifest_path, out_dir, num_speakers)

            logger.info(
                "Running NeMo diarization via %s (device=%s)...",
                self._diarizer_name or "unknown",
                cfg.device,
            )
            try:
                diarizer = self._diarizer_ctor(cfg=cfg)
                diarizer.diarize()
            except Exception as exc:  # pylint: disable=broad-except
                if self._diarizer_name != "NeuralDiarizer":
                    raise
                try:
                    from nemo.collections.asr.models import ClusteringDiarizer  # type: ignore
                except Exception:
                    raise
                logger.warning(
                    "NeuralDiarizer failed (%s). Retrying with ClusteringDiarizer.",
                    exc,
                )
                diarizer = ClusteringDiarizer(cfg=cfg)
                diarizer.diarize()

            rttm_path = out_dir / "pred_rttms" / f"{prepared_audio.stem}.rttm"
            if not rttm_path.exists():
                pred_rttm_dir = out_dir / "pred_rttms"
                candidates = sorted(pred_rttm_dir.glob("*.rttm")) if pred_rttm_dir.exists() else []
                if candidates:
                    rttm_path = candidates[0]
            return self._parse_rttm(rttm_path)
