from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: dict[str, Any] = {
    "cleanup_system_prompt": (
        "You are a senior Russian/English transcript editor for technical interviews.\n"
        "Primary goal: produce clean, correct, concise transcript WITHOUT adding new facts.\n"
        "You must preserve meaning, chronology, and speaker ownership."
    ),
    "cleanup_user_prompt_template": (
        "Language hint: {language_hint}\n"
        "Format: {output_format}\n"
        "Speaker label replacements (apply exactly): {speaker_label_replacements}\n"
        "Normalization map (apply aggressively when context matches): {normalization_hints}\n\n"
        "Edit rules:\n"
        "1) Fix spelling, punctuation, casing, grammar.\n"
        "2) Fix obvious ASR/OCR mistakes and malformed terms (examples: QI->QA, Key->QA, sted->SDET when context is QA interview).\n"
        "3) Remove filler, repetitions, stutters, interjections, verbal noise (e.g. 'nu', 'kak by', duplicated fragments) while preserving key meaning.\n"
        "4) Keep technical terms correct and consistent (REST API, CI/CD, GitLab CI, RabbitMQ, Kafka, QA Lead, SDET, Automation QA, Vue.js, Selenium, XPath).\n"
        "5) If dialogue format, keep one line per speaker: \"Speaker: text\".\n"
        "6) Speaker labels policy: use only provided replacements; if replacement map is empty or a speaker has no mapping, keep original labels exactly (User1, User2, ...); never invent speaker names.\n"
        "7) Do not invent details. If uncertain, keep original meaning conservatively.\n"
        "8) Return ONLY the edited transcript text, no explanations.\n\n"
        "Transcript chunk:\n"
        "{transcript_chunk}"
    ),
    "summary_chunk_system_prompt": (
        "You are a transcript analyst. Summarize accurately, no hallucinations. Keep source language."
    ),
    "summary_chunk_user_prompt_template": (
        "Language hint: {language_hint}\n"
        "Chunk: {chunk_index}/{chunk_total}\n"
        "Provide concise markdown:\n"
        "1) Topic (one line)\n"
        "2) Key points (3-8 bullets)\n"
        "3) Decisions/actions if present\n\n"
        "Transcript chunk:\n"
        "{transcript_chunk}"
    ),
    "summary_merge_system_prompt": (
        "You merge partial transcript summaries into one final summary. Keep only supported facts."
    ),
    "summary_merge_user_prompt_template": (
        "Language hint: {language_hint}\n"
        "Merge chunk summaries into one final concise markdown summary with sections:\n"
        "1) Topic\n"
        "2) Key points (5-12 bullets)\n"
        "3) Decisions/actions\n\n"
        "Chunk summaries:\n"
        "{chunk_summaries}"
    ),
    "normalization_map": {
        "QI-\u043b\u0438\u0442": "QA Lead",
        "qi-\u043b\u0438\u0442": "QA Lead",
        "QLED": "QA Lead",
        "qled": "QA Lead",
        "Automation Key": "Automation QA",
        "manual key": "manual QA",
        "\u0441\u0442\u0435\u0434-\u0438\u043d\u0436\u0435\u043d\u0435\u0440": "SDET-\u0438\u043d\u0436\u0435\u043d\u0435\u0440",
        "ZDET-\u0438\u043d\u0436\u0435\u043d\u0435\u0440": "SDET-\u0438\u043d\u0436\u0435\u043d\u0435\u0440",
        "\u0443\u0437\u0434\u0435\u0442\u0430": "SDET",
    },
}


@dataclass
class PostprocessReport:
    applied: bool
    method: str
    renamed_speakers: dict[str, str]
    error: str = ""


@dataclass
class DebugTextReport:
    cleanup_method: str
    summary_method: str
    error: str = ""


class TextPostProcessor:
    def __init__(self) -> None:
        self.enabled = self._env_bool("TEXT_POSTPROCESS_ENABLED", True)
        self.provider = os.getenv("TEXT_POSTPROCESS_PROVIDER", "lmstudio").strip().lower() or "lmstudio"
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").strip().rstrip("/")
        if not self.base_url:
            self.base_url = "http://127.0.0.1:1234"
        self.model = os.getenv("LMSTUDIO_MODEL", "").strip()
        self.timeout_sec = self._env_float("LMSTUDIO_TIMEOUT_SEC", 600.0)
        self.temperature = self._env_float("LMSTUDIO_TEMPERATURE", 0.0)
        self.max_chunk_chars = int(os.getenv("TEXT_POSTPROCESS_MAX_CHARS_PER_CHUNK", "2500"))
        self.max_chunk_chars = max(1000, min(self.max_chunk_chars, 20000))
        self.summary_chunk_chars = int(
            os.getenv("TEXT_SUMMARY_MAX_CHARS_PER_CHUNK", str(min(3500, self.max_chunk_chars)))
        )
        self.summary_chunk_chars = max(1000, min(self.summary_chunk_chars, 20000))
        self.request_retries = int(os.getenv("LMSTUDIO_REQUEST_RETRIES", "3"))
        self.request_retries = max(1, min(self.request_retries, 8))
        self.retry_backoff_sec = self._env_float("LMSTUDIO_RETRY_BACKOFF_SEC", 1.5)
        self.retry_backoff_sec = max(0.2, min(self.retry_backoff_sec, 10.0))
        self.min_resplit_chars = int(os.getenv("TEXT_POSTPROCESS_MIN_RESPLIT_CHARS", "1200"))
        self.min_resplit_chars = max(600, min(self.min_resplit_chars, 8000))
        self.gemini_api_keys = self._load_gemini_api_keys()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"
        self.gemini_timeout_sec = self._env_float("GEMINI_TIMEOUT_SEC", self.timeout_sec)
        self.gemini_timeout_sec = max(5.0, min(self.gemini_timeout_sec, 600.0))
        self.gemini_fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "whisper").strip().lower() or "whisper"
        prompts_file_value = os.getenv("LMSTUDIO_PROMPTS_FILE", "").strip()
        if prompts_file_value:
            prompts_path = Path(prompts_file_value)
            if not prompts_path.is_absolute():
                prompts_path = Path(__file__).resolve().parent / prompts_path
            self.prompts_file = prompts_path
        else:
            self.prompts_file = Path(__file__).with_name("lmstudio_prompts.json")
        self.prompts = self._load_prompts_profile(self.prompts_file)
        self.normalization_map = self._extract_normalization_map(self.prompts)
        self._resolved_model_cache: str | None = None
        self._resolved_api_base_url: str | None = None

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw.strip())
        except ValueError:
            return default

    @staticmethod
    def _env_multi_values(raw: str) -> list[str]:
        if not raw:
            return []
        normalized = raw.replace("\r", "\n").replace(";", ",").replace("\n", ",")
        parts = [item.strip() for item in normalized.split(",")]
        return [item for item in parts if item]

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        msg = str(exc).lower()
        return "timed out" in msg or "timeout" in msg

    @staticmethod
    def _safe_text(value: object, default: str = "") -> str:
        if not isinstance(value, str):
            return default
        normalized = value.strip()
        return normalized if normalized else default

    def _load_prompts_profile(self, file_path: Path) -> dict[str, Any]:
        profile: dict[str, Any] = dict(DEFAULT_PROMPTS)
        if not file_path.exists():
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(
                    json.dumps(DEFAULT_PROMPTS, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Created default LM Studio prompts file: %s", file_path)
            except OSError as exc:
                logger.warning("Could not create prompts file at %s: %s", file_path, exc)
            return profile

        try:
            raw = file_path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load LM Studio prompts file %s: %s", file_path, exc)
            return profile

        if not isinstance(loaded, dict):
            logger.warning("LM Studio prompts file must contain JSON object: %s", file_path)
            return profile

        for key, default_value in DEFAULT_PROMPTS.items():
            if key not in loaded:
                continue
            value = loaded[key]
            if isinstance(default_value, str):
                text_value = self._safe_text(value, default_value)
                profile[key] = text_value
                continue
            if isinstance(default_value, dict) and isinstance(value, dict):
                merged: dict[str, str] = {}
                for src, dst in value.items():
                    src_text = self._safe_text(src, "")
                    dst_text = self._safe_text(dst, "")
                    if src_text and dst_text:
                        merged[src_text] = dst_text
                profile[key] = merged
                continue
            profile[key] = value
        return profile

    @staticmethod
    def _extract_normalization_map(profile: dict[str, Any]) -> dict[str, str]:
        raw_map = profile.get("normalization_map")
        if not isinstance(raw_map, dict):
            return {}
        normalized: dict[str, str] = {}
        for src, dst in raw_map.items():
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            src_key = src.strip()
            dst_value = dst.strip()
            if src_key and dst_value:
                normalized[src_key] = dst_value
        return normalized

    def _load_gemini_api_keys(self) -> list[str]:
        raw_list = os.getenv("GEMINI_API_KEYS", "").strip()
        keys = self._env_multi_values(raw_list)
        deduped: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    @staticmethod
    def _apply_normalization_map(text: str, replacements: dict[str, str]) -> str:
        if not text or not replacements:
            return text
        updated = text
        for source, target in replacements.items():
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            updated = pattern.sub(target, updated)
        return updated

    def _candidate_api_base_urls(self) -> list[str]:
        base = self.base_url.rstrip("/")
        candidates: list[str] = []
        if base.endswith("/api/v1") or base.endswith("/v1"):
            candidates.append(base)
        else:
            candidates.append(f"{base}/v1")
            candidates.append(f"{base}/api/v1")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _split_text_chunks(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        lines = text.splitlines(keepends=True)
        chunks: list[str] = []
        current = ""
        for line in lines:
            if len(current) + len(line) <= max_chars:
                current += line
                continue
            if current:
                chunks.append(current)
                current = ""
            if len(line) <= max_chars:
                current = line
                continue
            # Hard split for extremely long lines.
            start = 0
            while start < len(line):
                end = min(start + max_chars, len(line))
                part = line[start:end]
                if len(part) == max_chars and end < len(line):
                    split_at = part.rfind(" ")
                    if split_at > max_chars // 2:
                        end = start + split_at
                        part = line[start:end]
                chunks.append(part)
                start = end
        if current:
            chunks.append(current)
        return chunks
    @staticmethod
    def _extract_speaker_name_map(text: str) -> dict[str, str]:
        """
        Heuristic extraction from lines like:
        User1: Привет, меня зовут Иван
        User2: Hi, my name is John
        """
        patterns = [
            re.compile(
                r"\b\u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442\s+([A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451-]{2,40})",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b\u044d\u0442\u043e\s+([A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451-]{2,40})",
                re.IGNORECASE,
            ),
            re.compile(r"\bmy name is\s+([A-Za-z-]{2,40})", re.IGNORECASE),
            re.compile(r"\bi am\s+([A-Za-z-]{2,40})", re.IGNORECASE),
            re.compile(r"\bi'm\s+([A-Za-z-]{2,40})", re.IGNORECASE),
        ]
        result: dict[str, str] = {}
        for line in text.splitlines():
            match = re.match(r"^(User\d+):\s*(.+)$", line.strip())
            if not match:
                continue
            speaker = match.group(1)
            payload = match.group(2)
            if speaker in result:
                continue
            for pattern in patterns:
                m = pattern.search(payload)
                if not m:
                    continue
                name = m.group(1).strip(".,!?;: ")
                if not name:
                    continue
                result[speaker] = name
                break
        return result

    @staticmethod
    def _apply_speaker_name_map(text: str, name_map: dict[str, str]) -> str:
        if not name_map:
            return text
        lines_out: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^(User\d+):\s*(.*)$", line)
            if not m:
                lines_out.append(line)
                continue
            speaker = m.group(1)
            payload = m.group(2)
            speaker_name = name_map.get(speaker)
            if speaker_name:
                lines_out.append(f"{speaker_name}: {payload}")
            else:
                lines_out.append(line)
        return "\n".join(lines_out)

    @staticmethod
    def _basic_cleanup(text: str) -> str:
        # Conservative fallback cleanup when LLM is unavailable.
        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"([,.;!?])([^\s\n])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        return cleaned

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
        *,
        service_name: str = "LM Studio",
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        total_attempts = max(1, self.request_retries)
        request_timeout = timeout_sec or self.timeout_sec
        for attempt in range(1, total_attempts + 1):
            try:
                data = None
                if payload is not None:
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                payload_bytes = len(data) if data is not None else 0
                logger.info(
                    "%s request start: %s %s (attempt %s/%s, timeout=%.1fs, payload_bytes=%s)",
                    service_name,
                    method,
                    url,
                    attempt,
                    total_attempts,
                    request_timeout,
                    payload_bytes,
                )
                request_headers = {"Content-Type": "application/json"}
                if headers:
                    request_headers.update(headers)
                req = urllib_request.Request(
                    url=url,
                    data=data,
                    method=method,
                    headers=request_headers,
                )
                request_started_at = time.perf_counter()
                heartbeat_stop = threading.Event()

                def _heartbeat() -> None:
                    while not heartbeat_stop.wait(30.0):
                        elapsed = time.perf_counter() - request_started_at
                        logger.info(
                            "%s request in progress: %s %s (attempt %s/%s, elapsed=%.1fs)",
                            service_name,
                            method,
                            url,
                            attempt,
                            total_attempts,
                            elapsed,
                        )

                heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
                heartbeat_thread.start()
                try:
                    with urllib_request.urlopen(req, timeout=request_timeout) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                finally:
                    heartbeat_stop.set()
                elapsed = time.perf_counter() - request_started_at
                logger.info(
                    "%s request done: %s %s (attempt %s/%s, elapsed=%.1fs)",
                    service_name,
                    method,
                    url,
                    attempt,
                    total_attempts,
                    elapsed,
                )
                return json.loads(body)
            except (
                TimeoutError,
                urllib_error.URLError,
                urllib_error.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_exc = exc
                if attempt >= total_attempts:
                    break
                if isinstance(exc, urllib_error.HTTPError):
                    code = int(getattr(exc, "code", 0) or 0)
                    # Retry only transient HTTP statuses.
                    if code and code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                        break
                sleep_for = self.retry_backoff_sec * attempt
                logger.warning(
                    "%s request failed (attempt %s/%s) for %s %s: %s. Retrying in %.1fs",
                    service_name,
                    attempt,
                    total_attempts,
                    method,
                    url,
                    exc,
                    sleep_for,
                )
                time.sleep(sleep_for)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LM Studio request failed unexpectedly without exception.")

    def _fetch_models_with_base_url(
        self,
        timeout_sec: float | None = None,
    ) -> tuple[str, list[str]]:
        candidates = self._candidate_api_base_urls()
        if self._resolved_api_base_url:
            preferred = self._resolved_api_base_url
            candidates = [preferred] + [c for c in candidates if c.lower() != preferred.lower()]

        attempts: list[str] = []
        last_exc: Exception | None = None

        for api_base_url in candidates:
            attempts.append(api_base_url)
            try:
                models_resp = self._request_json(
                    "GET",
                    f"{api_base_url}/models",
                    timeout_sec=timeout_sec,
                )
            except (
                RuntimeError,
                TimeoutError,
                urllib_error.URLError,
                urllib_error.HTTPError,
                json.JSONDecodeError,
            ) as exc:
                last_exc = exc
                continue

            data = models_resp.get("data") or []
            model_ids = [str(item.get("id") or "").strip() for item in data]
            model_ids = [mid for mid in model_ids if mid]
            self._resolved_api_base_url = api_base_url
            return api_base_url, model_ids

        attempts_text = ", ".join(attempts) if attempts else "(none)"
        if last_exc is not None:
            raise RuntimeError(
                f"LM Studio endpoint is unavailable. Tried: {attempts_text}. Last error: {last_exc}"
            ) from last_exc
        raise RuntimeError(f"LM Studio endpoint is unavailable. Tried: {attempts_text}.")

    def _resolve_api_base_url(self) -> str:
        if self._resolved_api_base_url:
            return self._resolved_api_base_url
        api_base_url, _ = self._fetch_models_with_base_url(timeout_sec=min(self.timeout_sec, 10.0))
        return api_base_url

    def _resolve_model_id(self) -> str:
        if self.model:
            return self.model
        if self._resolved_model_cache:
            return self._resolved_model_cache
        api_base_url, model_ids = self._fetch_models_with_base_url()
        if not model_ids:
            raise RuntimeError(f"LM Studio returned empty /models list at {api_base_url}.")
        model_id = model_ids[0]
        if not model_id:
            raise RuntimeError("LM Studio /models response has no model id.")
        self._resolved_model_cache = model_id
        return model_id

    def runtime_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "enabled": self.enabled,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_base_url": self._resolved_api_base_url or "",
            "prompts_file": str(self.prompts_file),
            "normalization_entries": len(self.normalization_map),
            "timeout_sec": self.timeout_sec,
            "request_retries": self.request_retries,
            "chunk_chars": self.max_chunk_chars,
            "summary_chunk_chars": self.summary_chunk_chars,
            "configured_model": self.model or "(auto)",
            "effective_model": self.model or self._resolved_model_cache or "",
            "gemini_model": self.gemini_model,
            "gemini_api_key_set": bool(self.gemini_api_keys),
            "gemini_api_keys_count": len(self.gemini_api_keys),
            "gemini_timeout_sec": self.gemini_timeout_sec,
            "gemini_fallback_model": self.gemini_fallback_model,
            "available": False,
            "models": [],
            "error": "",
        }
        if not self.enabled:
            return status
        if self.provider != "lmstudio":
            status["error"] = f"Unsupported provider: {self.provider}"
            return status
        try:
            api_base_url, model_ids = self._fetch_models_with_base_url(timeout_sec=min(self.timeout_sec, 8.0))
            status["api_base_url"] = api_base_url
            status["models"] = model_ids
            status["available"] = True
            if not self.model:
                if model_ids:
                    self._resolved_model_cache = model_ids[0]
                    status["effective_model"] = model_ids[0]
            else:
                status["effective_model"] = self.model
        except (RuntimeError, TimeoutError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
            status["error"] = str(exc)
        return status

    @staticmethod
    def _is_gemini_quota_error(exc: Exception) -> bool:
        message = str(exc).lower()
        quota_markers = {
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
            "429",
        }
        return any(marker in message for marker in quota_markers)

    @staticmethod
    def _is_gemini_auth_error(exc: Exception) -> bool:
        message = str(exc).lower()
        auth_markers = {
            "unauthorized",
            "permission denied",
            "invalid api key",
            "api key not valid",
            "403",
            "401",
        }
        return any(marker in message for marker in auth_markers)

    def _gemini_endpoint_url(self, model_name: str | None = None) -> str:
        model_id = (model_name or self.gemini_model).strip() or self.gemini_model
        model = urllib_parse.quote(model_id, safe="")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _gemini_model_candidates(self) -> list[str]:
        configured = (self.gemini_model or "").strip()
        if not configured:
            return ["gemini-2.5-flash-lite"]
        candidates = [configured]
        fallback_map = {
            "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
            "gemini-3.1-pro": "gemini-3.1-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
        }
        alt = fallback_map.get(configured)
        if alt and alt not in candidates:
            candidates.append(alt)
        return candidates

    @staticmethod
    def _extract_gemini_text(response: dict[str, Any]) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned empty candidates.")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        collected: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "").strip()
            if text:
                collected.append(text)
        merged = "\n".join(collected).strip()
        if not merged:
            raise RuntimeError("Gemini returned empty text content.")
        return merged

    def _generate_with_gemini(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        if not self.gemini_api_keys:
            raise RuntimeError("GEMINI_API_KEYS is not configured.")
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": float(max(0.0, min(temperature, 1.0)))},
        }
        model_candidates = self._gemini_model_candidates()
        total_keys = len(self.gemini_api_keys)
        last_exc: Exception | None = None
        for model_idx, model_name in enumerate(model_candidates, start=1):
            for idx, key in enumerate(self.gemini_api_keys, start=1):
                try:
                    response = self._request_json(
                        "POST",
                        self._gemini_endpoint_url(model_name=model_name),
                        payload,
                        timeout_sec=self.gemini_timeout_sec,
                        service_name="Gemini",
                        headers={"x-goog-api-key": key},
                    )
                    if idx > 1:
                        logger.info("Gemini succeeded with backup API key #%s/%s.", idx, total_keys)
                    if model_idx > 1:
                        logger.info(
                            "Gemini succeeded with fallback model id '%s' (configured '%s').",
                            model_name,
                            self.gemini_model,
                        )
                    return self._extract_gemini_text(response)
                except (RuntimeError, TimeoutError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
                    last_exc = exc
                    if self._is_gemini_quota_error(exc):
                        if idx < total_keys:
                            logger.warning(
                                "Gemini key #%s/%s hit quota/rate limit. Switching to next key.",
                                idx,
                                total_keys,
                            )
                            continue
                        break
                    if self._is_gemini_auth_error(exc):
                        if idx < total_keys:
                            logger.warning(
                                "Gemini key #%s/%s failed auth. Switching to next key.",
                                idx,
                                total_keys,
                            )
                            continue
                        break
                    if isinstance(exc, urllib_error.HTTPError) and int(getattr(exc, "code", 0) or 0) == 404:
                        logger.warning(
                            "Gemini model id '%s' returned 404. Trying next model id if configured.",
                            model_name,
                        )
                        break
                    # Non key-specific error (network/server), don't rotate blindly.
                    break
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini request failed unexpectedly without exception.")

    def _rewrite_chunk_with_lmstudio(
        self,
        chunk: str,
        language_hint: str,
        output_format: str,
        speaker_name_map: dict[str, str],
    ) -> str:
        model_id = self._resolve_model_id()
        system_prompt = self._safe_text(
            self.prompts.get("cleanup_system_prompt"),
            str(DEFAULT_PROMPTS["cleanup_system_prompt"]),
        )
        map_hint = json.dumps(speaker_name_map, ensure_ascii=False) if speaker_name_map else "{}"
        normalization_hint = json.dumps(self.normalization_map, ensure_ascii=False) if self.normalization_map else "{}"
        user_template = self._safe_text(
            self.prompts.get("cleanup_user_prompt_template"),
            str(DEFAULT_PROMPTS["cleanup_user_prompt_template"]),
        )
        try:
            user_prompt = user_template.format(
                language_hint=language_hint,
                output_format=output_format,
                speaker_label_replacements=map_hint,
                normalization_hints=normalization_hint,
                transcript_chunk=chunk,
            )
        except KeyError as exc:
            logger.warning("Invalid placeholder in cleanup prompt template: %s", exc)
            user_prompt = str(DEFAULT_PROMPTS["cleanup_user_prompt_template"]).format(
                language_hint=language_hint,
                output_format=output_format,
                speaker_label_replacements=map_hint,
                normalization_hints=normalization_hint,
                transcript_chunk=chunk,
            )
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        api_base_url = self._resolve_api_base_url()
        resp = self._request_json("POST", f"{api_base_url}/chat/completions", payload)
        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError("LM Studio returned empty choices.")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "").strip()
        if not content:
            raise RuntimeError("LM Studio returned empty content.")
        return content

    def _rewrite_chunk_with_gemini(
        self,
        chunk: str,
        language_hint: str,
        output_format: str,
        speaker_name_map: dict[str, str],
    ) -> str:
        system_prompt = self._safe_text(
            self.prompts.get("cleanup_system_prompt"),
            str(DEFAULT_PROMPTS["cleanup_system_prompt"]),
        )
        map_hint = json.dumps(speaker_name_map, ensure_ascii=False) if speaker_name_map else "{}"
        normalization_hint = json.dumps(self.normalization_map, ensure_ascii=False) if self.normalization_map else "{}"
        user_template = self._safe_text(
            self.prompts.get("cleanup_user_prompt_template"),
            str(DEFAULT_PROMPTS["cleanup_user_prompt_template"]),
        )
        try:
            user_prompt = user_template.format(
                language_hint=language_hint,
                output_format=output_format,
                speaker_label_replacements=map_hint,
                normalization_hints=normalization_hint,
                transcript_chunk=chunk,
            )
        except KeyError as exc:
            logger.warning("Invalid placeholder in cleanup prompt template: %s", exc)
            user_prompt = str(DEFAULT_PROMPTS["cleanup_user_prompt_template"]).format(
                language_hint=language_hint,
                output_format=output_format,
                speaker_label_replacements=map_hint,
                normalization_hints=normalization_hint,
                transcript_chunk=chunk,
            )
        return self._generate_with_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

    def _rewrite_chunk_resilient(
        self,
        chunk: str,
        language_hint: str,
        output_format: str,
        speaker_name_map: dict[str, str],
        depth: int = 0,
    ) -> str:
        try:
            return self._rewrite_chunk_with_lmstudio(
                chunk=chunk,
                language_hint=language_hint,
                output_format=output_format,
                speaker_name_map=speaker_name_map,
            )
        except (RuntimeError, TimeoutError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
            if (not self._is_timeout_error(exc)) or len(chunk) <= self.min_resplit_chars or depth >= 5:
                raise
            next_max = max(self.min_resplit_chars, len(chunk) // 2)
            subchunks = self._split_text_chunks(chunk, next_max)
            if len(subchunks) <= 1:
                raise
            logger.warning(
                "LM Studio timed out on chunk (len=%s, depth=%s). Splitting into %s smaller chunks.",
                len(chunk),
                depth,
                len(subchunks),
            )
            rewritten_parts: list[str] = []
            for sub in subchunks:
                rewritten_parts.append(
                    self._rewrite_chunk_resilient(
                        chunk=sub,
                        language_hint=language_hint,
                        output_format=output_format,
                        speaker_name_map=speaker_name_map,
                        depth=depth + 1,
                    )
                )
            return "\n".join(part.strip("\n") for part in rewritten_parts).strip()

    def _summarize_text_with_gemini(self, text: str, language_hint: str) -> str:
        started_at = time.perf_counter()
        chunks = self._split_text_chunks(text, self.summary_chunk_chars)
        logger.info(
            "Gemini summary started: chars=%s, chunk_max=%s, chunks=%s, language=%s",
            len(text),
            self.summary_chunk_chars,
            len(chunks),
            language_hint,
        )
        chunk_summaries: list[str] = []
        chunk_system_prompt = self._safe_text(
            self.prompts.get("summary_chunk_system_prompt"),
            str(DEFAULT_PROMPTS["summary_chunk_system_prompt"]),
        )
        chunk_user_template = self._safe_text(
            self.prompts.get("summary_chunk_user_prompt_template"),
            str(DEFAULT_PROMPTS["summary_chunk_user_prompt_template"]),
        )

        for idx, chunk in enumerate(chunks, start=1):
            chunk_started_at = time.perf_counter()
            logger.info(
                "Gemini summary chunk %s/%s started (len=%s).",
                idx,
                len(chunks),
                len(chunk),
            )
            try:
                user_prompt = chunk_user_template.format(
                    language_hint=language_hint,
                    chunk_index=idx,
                    chunk_total=len(chunks),
                    transcript_chunk=chunk,
                )
            except KeyError as exc:
                logger.warning("Invalid placeholder in summary chunk template: %s", exc)
                user_prompt = str(DEFAULT_PROMPTS["summary_chunk_user_prompt_template"]).format(
                    language_hint=language_hint,
                    chunk_index=idx,
                    chunk_total=len(chunks),
                    transcript_chunk=chunk,
                )
            content = self._generate_with_gemini(
                system_prompt=chunk_system_prompt,
                user_prompt=user_prompt,
                temperature=min(self.temperature, 0.2),
            )
            chunk_summaries.append(content)
            logger.info(
                "Gemini summary chunk %s/%s finished in %.1fs (output_len=%s).",
                idx,
                len(chunks),
                time.perf_counter() - chunk_started_at,
                len(content),
            )

        if len(chunk_summaries) == 1:
            logger.info(
                "Gemini summary finished in %.1fs (single chunk, output_chars=%s).",
                time.perf_counter() - started_at,
                len(chunk_summaries[0]),
            )
            return chunk_summaries[0]

        combined = "\n\n".join(f"Chunk {i + 1} summary:\n{part}" for i, part in enumerate(chunk_summaries))
        merge_system_prompt = self._safe_text(
            self.prompts.get("summary_merge_system_prompt"),
            str(DEFAULT_PROMPTS["summary_merge_system_prompt"]),
        )
        merge_user_template = self._safe_text(
            self.prompts.get("summary_merge_user_prompt_template"),
            str(DEFAULT_PROMPTS["summary_merge_user_prompt_template"]),
        )
        try:
            merge_user_prompt = merge_user_template.format(
                language_hint=language_hint,
                chunk_summaries=combined,
            )
        except KeyError as exc:
            logger.warning("Invalid placeholder in summary merge template: %s", exc)
            merge_user_prompt = str(DEFAULT_PROMPTS["summary_merge_user_prompt_template"]).format(
                language_hint=language_hint,
                chunk_summaries=combined,
            )
        logger.info(
            "Gemini summary merge started for %s partial summaries (combined_len=%s).",
            len(chunk_summaries),
            len(combined),
        )
        merge_started_at = time.perf_counter()
        final_text = self._generate_with_gemini(
            system_prompt=merge_system_prompt,
            user_prompt=merge_user_prompt,
            temperature=min(self.temperature, 0.2),
        )
        logger.info(
            "Gemini summary merge finished in %.1fs (output_len=%s).",
            time.perf_counter() - merge_started_at,
            len(final_text),
        )
        logger.info(
            "Gemini summary finished in %.1fs (output_chars=%s).",
            time.perf_counter() - started_at,
            len(final_text),
        )
        return final_text

    def _summarize_text_with_lmstudio(self, text: str, language_hint: str) -> str:
        started_at = time.perf_counter()
        model_id = self._resolve_model_id()
        api_base_url = self._resolve_api_base_url()

        chunks = self._split_text_chunks(text, self.summary_chunk_chars)
        logger.info(
            "Text summary started: chars=%s, chunk_max=%s, chunks=%s, language=%s",
            len(text),
            self.summary_chunk_chars,
            len(chunks),
            language_hint,
        )
        chunk_summaries: list[str] = []
        chunk_system_prompt = self._safe_text(
            self.prompts.get("summary_chunk_system_prompt"),
            str(DEFAULT_PROMPTS["summary_chunk_system_prompt"]),
        )
        chunk_user_template = self._safe_text(
            self.prompts.get("summary_chunk_user_prompt_template"),
            str(DEFAULT_PROMPTS["summary_chunk_user_prompt_template"]),
        )

        for idx, chunk in enumerate(chunks, start=1):
            chunk_started_at = time.perf_counter()
            logger.info(
                "Summary chunk %s/%s started (len=%s).",
                idx,
                len(chunks),
                len(chunk),
            )
            try:
                user_prompt = chunk_user_template.format(
                    language_hint=language_hint,
                    chunk_index=idx,
                    chunk_total=len(chunks),
                    transcript_chunk=chunk,
                )
            except KeyError as exc:
                logger.warning("Invalid placeholder in summary chunk template: %s", exc)
                user_prompt = str(DEFAULT_PROMPTS["summary_chunk_user_prompt_template"]).format(
                    language_hint=language_hint,
                    chunk_index=idx,
                    chunk_total=len(chunks),
                    transcript_chunk=chunk,
                )
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": chunk_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": min(self.temperature, 0.2),
            }
            resp = self._request_json("POST", f"{api_base_url}/chat/completions", payload)
            choices = resp.get("choices") or []
            if not choices:
                raise RuntimeError("LM Studio returned empty choices for summary.")
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            if not content:
                raise RuntimeError("LM Studio returned empty summary content.")
            chunk_summaries.append(content)
            logger.info(
                "Summary chunk %s/%s finished in %.1fs (output_len=%s).",
                idx,
                len(chunks),
                time.perf_counter() - chunk_started_at,
                len(content),
            )

        if len(chunk_summaries) == 1:
            logger.info(
                "Text summary finished in %.1fs (single chunk, output_chars=%s).",
                time.perf_counter() - started_at,
                len(chunk_summaries[0]),
            )
            return chunk_summaries[0]

        combined = "\n\n".join(f"Chunk {i + 1} summary:\n{part}" for i, part in enumerate(chunk_summaries))
        merge_system_prompt = self._safe_text(
            self.prompts.get("summary_merge_system_prompt"),
            str(DEFAULT_PROMPTS["summary_merge_system_prompt"]),
        )
        merge_user_template = self._safe_text(
            self.prompts.get("summary_merge_user_prompt_template"),
            str(DEFAULT_PROMPTS["summary_merge_user_prompt_template"]),
        )
        try:
            merge_user_prompt = merge_user_template.format(
                language_hint=language_hint,
                chunk_summaries=combined,
            )
        except KeyError as exc:
            logger.warning("Invalid placeholder in summary merge template: %s", exc)
            merge_user_prompt = str(DEFAULT_PROMPTS["summary_merge_user_prompt_template"]).format(
                language_hint=language_hint,
                chunk_summaries=combined,
            )
        final_payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": merge_system_prompt},
                {"role": "user", "content": merge_user_prompt},
            ],
            "temperature": min(self.temperature, 0.2),
        }
        logger.info(
            "Summary merge started for %s partial summaries (combined_len=%s).",
            len(chunk_summaries),
            len(combined),
        )
        merge_started_at = time.perf_counter()
        final_resp = self._request_json("POST", f"{api_base_url}/chat/completions", final_payload)
        final_choices = final_resp.get("choices") or []
        if not final_choices:
            raise RuntimeError("LM Studio returned empty merged summary.")
        final_message = final_choices[0].get("message") or {}
        final_text = str(final_message.get("content") or "").strip()
        if not final_text:
            raise RuntimeError("LM Studio returned empty merged summary text.")
        logger.info(
            "Summary merge finished in %.1fs (output_len=%s).",
            time.perf_counter() - merge_started_at,
            len(final_text),
        )
        logger.info(
            "Text summary finished in %.1fs (output_chars=%s).",
            time.perf_counter() - started_at,
            len(final_text),
        )
        return final_text

    @staticmethod
    def _summarize_text_heuristic(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return "No content."
        sentences = re.split(r"(?<=[.!?])\s+", normalized)
        selected: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            if sentence in selected:
                continue
            selected.append(sentence)
            if len(selected) >= 8:
                break
        if not selected:
            selected = [normalized[:320] + ("..." if len(normalized) > 320 else "")]
        bullets = "\n".join(f"- {item}" for item in selected)
        return "Topic: transcript summary\n\nKey points:\n" + bullets

    def process_text(
        self,
        text: str,
        language_hint: str = "auto",
        output_format: str = "text",
        model_choice: str = "gemini",
    ) -> tuple[str, PostprocessReport]:
        started_at = time.perf_counter()
        if not text.strip():
            return text, PostprocessReport(applied=False, method="none", renamed_speakers={})
        if not self.enabled:
            return text, PostprocessReport(applied=False, method="disabled", renamed_speakers={})

        selected_model = (model_choice or "gemini").strip().lower()
        if selected_model not in {"gemini", "whisper", "oos20"}:
            selected_model = "gemini"
        logger.info(
            "Text cleanup started: model=%s, chars=%s, language=%s, format=%s, chunk_max=%s",
            selected_model,
            len(text),
            language_hint,
            output_format,
            self.max_chunk_chars,
        )

        if selected_model == "whisper":
            logger.info("Text cleanup skipped (model=whisper). Returning raw transcript text.")
            return text, PostprocessReport(applied=False, method="whisper", renamed_speakers={})

        speaker_name_map = self._extract_speaker_name_map(text)
        chunks = self._split_text_chunks(text, self.max_chunk_chars)
        logger.info(
            "Text cleanup split into %s chunk(s). Detected speaker aliases: %s",
            len(chunks),
            len(speaker_name_map),
        )

        out_chunks: list[str] = []
        try:
            if selected_model == "gemini" and not self.gemini_api_keys:
                error = "GEMINI_API_KEYS is not configured."
                if self.gemini_fallback_model == "oos20":
                    logger.warning("%s Falling back to oos20.", error)
                    return self.process_text(
                        text=text,
                        language_hint=language_hint,
                        output_format=output_format,
                        model_choice="oos20",
                    )
                logger.warning("%s Falling back to whisper mode.", error)
                return text, PostprocessReport(
                    applied=False,
                    method="whisper-fallback",
                    renamed_speakers=speaker_name_map,
                    error=error,
                )

            for idx, chunk in enumerate(chunks, start=1):
                chunk_started_at = time.perf_counter()
                logger.info(
                    "Cleanup chunk %s/%s started (len=%s).",
                    idx,
                    len(chunks),
                    len(chunk),
                )
                if selected_model == "gemini":
                    cleaned_chunk = self._rewrite_chunk_with_gemini(
                        chunk=chunk,
                        language_hint=language_hint,
                        output_format=output_format,
                        speaker_name_map=speaker_name_map,
                    )
                else:
                    cleaned_chunk = self._rewrite_chunk_resilient(
                        chunk=chunk,
                        language_hint=language_hint,
                        output_format=output_format,
                        speaker_name_map=speaker_name_map,
                    )
                if not cleaned_chunk.strip():
                    logger.warning("LLM returned empty chunk #%s, using original chunk.", idx)
                    cleaned_chunk = chunk
                out_chunks.append(cleaned_chunk)
                logger.info(
                    "Cleanup chunk %s/%s finished in %.1fs (output_len=%s).",
                    idx,
                    len(chunks),
                    time.perf_counter() - chunk_started_at,
                    len(cleaned_chunk),
                )
            cleaned = "\n".join(c.strip("\n") for c in out_chunks).strip()
            cleaned = self._apply_speaker_name_map(cleaned, speaker_name_map)
            cleaned = self._apply_normalization_map(cleaned, self.normalization_map)
            if not cleaned:
                cleaned = text
            logger.info(
                "Text cleanup finished via %s in %.1fs (output_chars=%s).",
                "Gemini" if selected_model == "gemini" else "LM Studio",
                time.perf_counter() - started_at,
                len(cleaned),
            )
            return cleaned, PostprocessReport(
                applied=cleaned != text,
                method="gemini" if selected_model == "gemini" else "lmstudio",
                renamed_speakers=speaker_name_map,
            )
        except (RuntimeError, TimeoutError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
            if selected_model == "gemini":
                logger.warning("Text post-processing via Gemini failed: %s", exc)
                if self._is_gemini_quota_error(exc):
                    if self.gemini_fallback_model == "oos20":
                        logger.warning("Gemini quota/rate limit detected. Falling back to oos20.")
                        return self.process_text(
                            text=text,
                            language_hint=language_hint,
                            output_format=output_format,
                            model_choice="oos20",
                        )
                    logger.warning("Gemini quota/rate limit detected. Falling back to whisper mode.")
                    return text, PostprocessReport(
                        applied=False,
                        method="whisper-fallback",
                        renamed_speakers=speaker_name_map,
                        error=str(exc),
                    )
                if self.gemini_fallback_model == "oos20":
                    logger.warning("Gemini failed. Falling back to oos20.")
                    return self.process_text(
                        text=text,
                        language_hint=language_hint,
                        output_format=output_format,
                        model_choice="oos20",
                    )
                return text, PostprocessReport(
                    applied=False,
                    method="whisper-fallback",
                    renamed_speakers=speaker_name_map,
                    error=str(exc),
                )
            logger.warning("Text post-processing via LM Studio failed: %s", exc)
            cleaned = self._basic_cleanup(text)
            cleaned = self._apply_speaker_name_map(cleaned, speaker_name_map)
            cleaned = self._apply_normalization_map(cleaned, self.normalization_map)
            logger.info(
                "Text cleanup fallback (heuristic) finished in %.1fs (output_chars=%s).",
                time.perf_counter() - started_at,
                len(cleaned),
            )
            return cleaned, PostprocessReport(
                applied=cleaned != text,
                method="heuristic-fallback",
                renamed_speakers=speaker_name_map,
                error=str(exc),
            )

    def process_debug_text(
        self,
        text: str,
        language_hint: str = "auto",
        model_choice: str = "gemini",
    ) -> tuple[str, str, DebugTextReport]:
        started_at = time.perf_counter()
        selected_model = (model_choice or "gemini").strip().lower()
        if selected_model not in {"gemini", "whisper", "oos20"}:
            selected_model = "gemini"
        logger.info(
            "Debug text processing started: chars=%s, language=%s, model=%s",
            len(text),
            language_hint,
            selected_model,
        )
        cleaned_text, cleanup_report = self.process_text(
            text=text,
            language_hint=language_hint,
            output_format="text",
            model_choice=selected_model,
        )
        logger.info(
            "Debug text cleanup step complete: method=%s, output_chars=%s",
            cleanup_report.method,
            len(cleaned_text),
        )

        if not cleaned_text.strip():
            logger.info("Debug text processing finished: empty text after cleanup.")
            return cleaned_text, "No content.", DebugTextReport(
                cleanup_method=cleanup_report.method,
                summary_method="none",
                error=cleanup_report.error,
            )

        summary_engine = selected_model
        if cleanup_report.method == "lmstudio":
            summary_engine = "oos20"
        if cleanup_report.method.startswith("whisper"):
            summary_engine = "whisper"
        if cleanup_report.method == "heuristic-fallback":
            summary_engine = "whisper"

        if summary_engine == "whisper":
            summary = self._summarize_text_heuristic(cleaned_text)
            logger.info(
                "Debug text summary finished via heuristic in %.1fs (summary_chars=%s).",
                time.perf_counter() - started_at,
                len(summary),
            )
            return cleaned_text, summary, DebugTextReport(
                cleanup_method=cleanup_report.method,
                summary_method="heuristic",
                error=cleanup_report.error,
            )

        try:
            if summary_engine == "gemini":
                summary = self._summarize_text_with_gemini(cleaned_text, language_hint)
            else:
                summary = self._summarize_text_with_lmstudio(cleaned_text, language_hint)
            logger.info(
                "Debug text processing finished via %s in %.1fs (summary_chars=%s).",
                "Gemini" if summary_engine == "gemini" else "LM Studio",
                time.perf_counter() - started_at,
                len(summary),
            )
            return cleaned_text, summary, DebugTextReport(
                cleanup_method=cleanup_report.method,
                summary_method="gemini" if summary_engine == "gemini" else "lmstudio",
                error=cleanup_report.error,
            )
        except (RuntimeError, TimeoutError, urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError) as exc:
            if summary_engine == "gemini":
                logger.warning("Text summary via Gemini failed: %s", exc)
                if self._is_gemini_quota_error(exc) and self.gemini_fallback_model == "oos20":
                    logger.warning("Gemini summary quota/rate limit detected. Falling back to oos20 summary.")
                    try:
                        summary = self._summarize_text_with_lmstudio(cleaned_text, language_hint)
                        return cleaned_text, summary, DebugTextReport(
                            cleanup_method=cleanup_report.method,
                            summary_method="lmstudio-fallback",
                            error=cleanup_report.error,
                        )
                    except (
                        RuntimeError,
                        TimeoutError,
                        urllib_error.URLError,
                        urllib_error.HTTPError,
                        json.JSONDecodeError,
                    ) as lm_exc:
                        logger.warning("oos20 summary fallback failed: %s", lm_exc)
                        summary = self._summarize_text_heuristic(cleaned_text)
                        merged_error = cleanup_report.error
                        if merged_error:
                            merged_error = f"{merged_error}; summary: {exc}; oos20 summary fallback: {lm_exc}"
                        else:
                            merged_error = f"summary: {exc}; oos20 summary fallback: {lm_exc}"
                        return cleaned_text, summary, DebugTextReport(
                            cleanup_method=cleanup_report.method,
                            summary_method="heuristic-fallback",
                            error=merged_error,
                        )
            else:
                logger.warning("Text summary via LM Studio failed: %s", exc)
            summary = self._summarize_text_heuristic(cleaned_text)
            merged_error = cleanup_report.error
            if merged_error:
                merged_error = f"{merged_error}; summary: {exc}"
            else:
                merged_error = f"summary: {exc}"
            logger.info(
                "Debug text processing finished with summary fallback in %.1fs (summary_chars=%s).",
                time.perf_counter() - started_at,
                len(summary),
            )
            return cleaned_text, summary, DebugTextReport(
                cleanup_method=cleanup_report.method,
                summary_method="heuristic-fallback",
                error=merged_error,
            )

