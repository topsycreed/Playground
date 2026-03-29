from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request as urllib_request

from nutrition_ai import (
    DEFAULT_MODEL_CANDIDATES,
    GEMINI_ENDPOINT,
    MODEL_ALIASES,
    GeminiNutritionAnalyzer,
    GeminiRequestError,
)


SYSTEM_PROMPT = (
    "You are an assistant that reads fitness tracker screenshots. "
    "Extract burned calories for today. Return JSON only."
)

USER_PROMPT = (
    "Read this fitness/activity app screenshot and extract active calories burned today. "
    "If uncertain, give best estimate from visible numbers. "
    "Use JSON exactly:\n"
    "{\n"
    '  "burned_kcal": 0,\n'
    '  "confidence": "low|medium|high",\n'
    '  "source_text": "string"\n'
    "}"
)


@dataclass
class ActivityEstimate:
    burned_kcal: float
    confidence: str
    source_text: str


class GeminiActivityAnalyzer:
    def __init__(self, api_keys: list[str], model: str, timeout_sec: float = 60.0) -> None:
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        self.model = (model or "gemini-3.1-flash-lite-preview").strip()
        self.timeout_sec = max(5.0, min(float(timeout_sec), 180.0))

    @classmethod
    def from_env(cls) -> "GeminiActivityAnalyzer":
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        for sep in (";", "\n"):
            raw_keys = raw_keys.replace(sep, ",")
        keys = [part.strip() for part in raw_keys.split(",") if part.strip()]
        model = (
            os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip()
            or "gemini-3.1-flash-lite-preview"
        )
        timeout_raw = os.getenv("GEMINI_TIMEOUT_SEC", "60").strip()
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = 60.0
        return cls(api_keys=keys, model=model, timeout_sec=timeout)

    def extract_activity(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> ActivityEstimate:
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEYS is empty.")
        if not image_bytes:
            raise RuntimeError("Image is empty.")

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": USER_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": mime_type or "image/jpeg",
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "maxOutputTokens": 256,
            },
        }

        response = self._generate_with_key_rotation(payload)
        text = GeminiNutritionAnalyzer._extract_text(response)
        parsed = GeminiNutritionAnalyzer._parse_json(text)
        return self._to_estimate(parsed)

    def _model_candidates(self) -> list[str]:
        configured = (self.model or "").strip()
        out: list[str] = []

        def push(model_id: str) -> None:
            value = (model_id or "").strip()
            if value and value not in out:
                out.append(value)

        if configured:
            push(MODEL_ALIASES.get(configured, configured))
            push(configured)
        for fallback in DEFAULT_MODEL_CANDIDATES:
            push(fallback)
        return out

    def _generate_with_key_rotation(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        total = len(self.api_keys)
        for model in self._model_candidates():
            endpoint = GEMINI_ENDPOINT.format(model=model)
            for idx, key in enumerate(self.api_keys, start=1):
                try:
                    return self._request_json(endpoint, payload, key)
                except GeminiRequestError as exc:
                    last_error = exc
                    if exc.status_code == 404:
                        break
                    if idx < total and self._can_try_next_key(exc):
                        continue
                    if self._can_try_next_key(exc):
                        break
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed with unknown error.")

    def _request_json(self, url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        # Reuse robust request implementation from nutrition analyzer.
        helper = GeminiNutritionAnalyzer(self.api_keys, self.model, self.timeout_sec)
        return helper._request_json(url, payload, api_key)

    @staticmethod
    def _can_try_next_key(error: GeminiRequestError) -> bool:
        if error.status_code in {401, 403, 429}:
            return True
        body = (error.body or "").lower()
        return any(token in body for token in ("quota", "rate limit", "resource_exhausted", "api key not valid"))

    @staticmethod
    def _to_estimate(payload: dict[str, Any]) -> ActivityEstimate:
        try:
            burned = float(payload.get("burned_kcal", 0.0) or 0.0)
        except (TypeError, ValueError):
            burned = 0.0
        burned = max(0.0, min(burned, 10000.0))

        confidence = str(payload.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"

        source_text = str(payload.get("source_text") or "").strip()[:200]
        return ActivityEstimate(
            burned_kcal=burned,
            confidence=confidence,
            source_text=source_text,
        )
