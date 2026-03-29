from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODEL_ALIASES = {
    "gemini-3-flash-preview": "gemini-2.5-flash",
    "gemini-2.0-flash": "gemini-2.5-flash",
}
DEFAULT_MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

SYSTEM_PROMPT = (
    "You are a nutrition assistant. Analyze a single food photo and estimate calories and macros. "
    "Return JSON only. Do not use markdown."
)

USER_PROMPT = (
    "Estimate total meal nutrition from this image. "
    "If unsure, provide the best estimate and include assumptions. "
    "Use this JSON shape exactly:\n"
    "{\n"
    '  "dish_name": "string",\n'
    '  "portion_grams": 0,\n'
    '  "calories_kcal": 0,\n'
    '  "protein_g": 0,\n'
    '  "fat_g": 0,\n'
    '  "carbs_g": 0,\n'
    '  "error_margin_percent": 0,\n'
    '  "confidence": "low|medium|high",\n'
    '  "assumptions": ["string"]\n'
    "}"
)

TEXT_USER_PROMPT = (
    "Estimate total meal nutrition from this text description. "
    "If quantity is unclear, assume a realistic portion and list assumptions. "
    "Use this JSON shape exactly:\n"
    "{\n"
    '  "dish_name": "string",\n'
    '  "portion_grams": 0,\n'
    '  "calories_kcal": 0,\n'
    '  "protein_g": 0,\n'
    '  "fat_g": 0,\n'
    '  "carbs_g": 0,\n'
    '  "error_margin_percent": 0,\n'
    '  "confidence": "low|medium|high",\n'
    '  "assumptions": ["string"]\n'
    "}"
)


@dataclass
class MacroEstimate:
    dish_name: str
    portion_grams: float
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    error_margin_percent: float
    confidence: str
    assumptions: list[str]


class GeminiRequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class GeminiNutritionAnalyzer:
    def __init__(self, api_keys: list[str], model: str, timeout_sec: float = 60.0) -> None:
        self.api_keys = [k.strip() for k in api_keys if k and k.strip()]
        self.model = (model or "gemini-3.1-flash-lite-preview").strip()
        self.timeout_sec = max(5.0, min(float(timeout_sec), 180.0))

    @classmethod
    def from_env(cls) -> "GeminiNutritionAnalyzer":
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

    def analyze_photo(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> MacroEstimate:
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEYS is empty.")
        if not image_bytes:
            raise RuntimeError("Image is empty.")
        payload = self._build_payload(image_bytes=image_bytes, mime_type=mime_type)
        response = self._generate_with_key_rotation(payload)
        text = self._extract_text(response)
        parsed = self._parse_json(text)
        return self._to_estimate(parsed)

    def analyze_description(self, description: str) -> MacroEstimate:
        if not self.api_keys:
            raise RuntimeError("GEMINI_API_KEYS is empty.")
        text = (description or "").strip()
        if len(text) < 3:
            raise RuntimeError("Meal description is too short.")
        payload = self._build_text_payload(description=text)
        response = self._generate_with_key_rotation(payload)
        model_text = self._extract_text(response)
        parsed = self._parse_json(model_text)
        return self._to_estimate(parsed)

    def _build_payload(self, image_bytes: bytes, mime_type: str) -> dict[str, Any]:
        return {
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
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "maxOutputTokens": 512,
            },
        }

    def _build_text_payload(self, description: str) -> dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": TEXT_USER_PROMPT},
                        {"text": f"Meal description: {description.strip()}"},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "maxOutputTokens": 512,
            },
        }

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
                        # Model ID is invalid/deprecated; try next model candidate.
                        break
                    if idx < total and self._can_try_next_key(exc):
                        continue
                    if self._can_try_next_key(exc):
                        # Key-related issue on final key, try next model candidate.
                        break
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed with unknown error.")

    def _request_json(self, url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except urllib_error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise GeminiRequestError(
                f"Gemini HTTP {exc.code}",
                status_code=int(exc.code),
                body=body_text,
            ) from exc
        except urllib_error.URLError as exc:
            raise GeminiRequestError(f"Gemini network error: {exc}") from exc
        except TimeoutError as exc:
            raise GeminiRequestError("Gemini timeout") from exc
        except json.JSONDecodeError as exc:
            raise GeminiRequestError(f"Gemini returned invalid JSON: {exc}") from exc

    @staticmethod
    def _can_try_next_key(error: GeminiRequestError) -> bool:
        if error.status_code in {401, 403, 429}:
            return True
        body = (error.body or "").lower()
        return any(token in body for token in ("quota", "rate limit", "resource_exhausted", "api key not valid"))

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            prompt_feedback = response.get("promptFeedback") or {}
            block_reason = str(prompt_feedback.get("blockReason") or "").strip()
            if block_reason:
                raise RuntimeError(f"Gemini blocked the request: {block_reason}")
            raise RuntimeError("Gemini returned no candidates.")
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        text_fragments: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            value = str(part.get("text") or "").strip()
            if value:
                text_fragments.append(value)
        merged = "\n".join(text_fragments).strip()
        if not merged:
            finish_reason = str((candidates[0] or {}).get("finishReason") or "").strip()
            if finish_reason:
                raise RuntimeError(f"Gemini returned empty text (finishReason={finish_reason}).")
            raise RuntimeError("Gemini returned empty text.")
        return merged

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
            try:
                data = json.loads(stripped)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if 0 <= start < end:
            candidate = stripped[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        raise RuntimeError("Could not parse JSON from Gemini response.")

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_margin_percent(value: Any, default: float = 20.0) -> float:
        margin = GeminiNutritionAnalyzer._to_float(value, default=default)
        return max(5.0, min(margin, 40.0))

    @classmethod
    def _to_estimate(cls, payload: dict[str, Any]) -> MacroEstimate:
        raw_assumptions = payload.get("assumptions") or []
        assumptions = [str(item).strip() for item in raw_assumptions if str(item).strip()]
        confidence = str(payload.get("confidence") or "medium").strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        return MacroEstimate(
            dish_name=str(payload.get("dish_name") or "Unknown dish").strip() or "Unknown dish",
            portion_grams=max(0.0, cls._to_float(payload.get("portion_grams"))),
            calories_kcal=max(0.0, cls._to_float(payload.get("calories_kcal"))),
            protein_g=max(0.0, cls._to_float(payload.get("protein_g"))),
            fat_g=max(0.0, cls._to_float(payload.get("fat_g"))),
            carbs_g=max(0.0, cls._to_float(payload.get("carbs_g"))),
            error_margin_percent=cls._normalize_margin_percent(
                payload.get("error_margin_percent"), default=20.0
            ),
            confidence=confidence,
            assumptions=assumptions[:5],
        )


def user_friendly_error(exc: Exception) -> str:
    if isinstance(exc, GeminiRequestError):
        code = int(exc.status_code or 0)
        body = (exc.body or "").lower()
        if code in {401, 403} or "api key not valid" in body:
            return "Проблема с Gemini API ключом. Проверь GEMINI_API_KEYS."
        if code == 404:
            return "Выбранная модель Gemini недоступна. Я переключу модель автоматически, попробуй снова."
        if code == 429 or "quota" in body or "rate limit" in body or "resource_exhausted" in body:
            return "Достигнут лимит Gemini (quota/rate limit). Попробуй позже."
        if code >= 500:
            return "Сервис Gemini временно недоступен (5xx). Попробуй снова через пару минут."
        return f"Ошибка Gemini (HTTP {code})."

    text = str(exc).strip().lower()
    if "timeout" in text:
        return "Gemini не ответил вовремя (timeout). Попробуй ещё раз."
    if "blocked" in text:
        return "Gemini заблокировал этот запрос. Попробуй другое фото или описание блюда текстом."
    if "empty text" in text or "no candidates" in text:
        return "Gemini не смог корректно сформировать ответ. Попробуй снова."
    return "Техническая ошибка анализа. Попробуй отправить фото ещё раз."
