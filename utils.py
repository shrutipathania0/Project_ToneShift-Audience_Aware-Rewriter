"""Utility functions for ToneShift."""
# Supports both local .env files and Streamlit Cloud Secrets Management.

from __future__ import annotations

import difflib
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import groq
from dotenv import load_dotenv
from groq import Groq

from prompts import (
    BACK_TRANSLATION_SYSTEM,
    MEANING_CHECK_SYSTEM,
    MODEL_NAME,
    QUALITY_SCORE_SYSTEM,
    REWRITE_SYSTEM_PROMPT,
    build_back_translation_prompt,
    build_meaning_check_prompt,
    build_quality_score_prompt,
    build_rewrite_user_prompt,
)

load_dotenv()

WORDS_PER_MINUTE = 200


@dataclass
class LLMResult:
    text: str
    raw: Any = None


@dataclass
class MeaningCheckResult:
    status: str
    confidence: int
    meaning_preservation_score: int
    explanation: str
    neutral_text: str


@dataclass
class QualityScores:
    meaning_preservation: int
    grammar: int
    readability: int
    tone_accuracy: int
    audience_match: int
    overall: int


class ToneShiftError(Exception):
    """Base application error with user-friendly message."""

    def __init__(self, message: str, category: str = "general"):
        super().__init__(message)
        self.message = message
        self.category = category


def get_api_key() -> Optional[str]:
    """Return the Groq API key.

    Resolution order:
    1. Environment variable / .env file  (local development)
    2. st.secrets["GROQ_API_KEY"]        (Streamlit Cloud deployment)
    """
    # 1. Try environment / .env
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    # 2. Try Streamlit Secrets (cloud deployment)
    try:
        import streamlit as st  # noqa: PLC0415
        key = st.secrets.get("GROQ_API_KEY", "").strip()
        if key:
            return key
    except Exception:  # streamlit not available or secret missing
        pass

    return None


def create_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def count_characters(text: str) -> int:
    return len(text or "")


def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b\w+\b", text))


def estimate_reading_time_minutes(text: str) -> float:
    words = count_words(text)
    if words == 0:
        return 0.0
    return round(words / WORDS_PER_MINUTE, 1)


def similarity_percentage(text_a: str, text_b: str) -> float:
    if not text_a and not text_b:
        return 100.0
    if not text_a or not text_b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    return round(ratio * 100, 1)


def highlight_word_differences(original: str, rewritten: str) -> tuple[str, str]:
    """Return HTML-highlighted versions of original and rewritten text."""
    original_words = original.split()
    rewritten_words = rewritten.split()

    matcher = difflib.SequenceMatcher(None, original_words, rewritten_words)

    original_parts: list[str] = []
    rewritten_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_chunk = " ".join(original_words[i1:i2])
        rew_chunk = " ".join(rewritten_words[j1:j2])

        if tag == "equal":
            original_parts.append(html.escape(orig_chunk))
            rewritten_parts.append(html.escape(rew_chunk))
        elif tag == "replace":
            original_parts.append(_wrap_span(orig_chunk, "diff-removed"))
            rewritten_parts.append(_wrap_span(rew_chunk, "diff-added"))
        elif tag == "delete":
            original_parts.append(_wrap_span(orig_chunk, "diff-removed"))
        elif tag == "insert":
            rewritten_parts.append(_wrap_span(rew_chunk, "diff-added"))

    return " ".join(original_parts), " ".join(rewritten_parts)


def _wrap_span(text: str, css_class: str) -> str:
    if not text:
        return ""
    return f'<span class="{css_class}">{html.escape(text)}</span>'


def _extract_response_text(response: Any) -> str:
    if response is None:
        raise ToneShiftError("The model returned an empty response.", "invalid_response")

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ToneShiftError("The model returned an invalid or empty response.", "invalid_response")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message else None
    if content is not None and str(content).strip():
        return str(content).strip()

    raise ToneShiftError("The model returned an invalid or empty response.", "invalid_response")


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _call_groq(
    client: Groq,
    *,
    system_instruction: str,
    user_prompt: str,
    temperature: float = 0.4,
    json_mode: bool = False,
    max_output_tokens: int = 8192,
) -> LLMResult:
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        text = _extract_response_text(response)
        if json_mode:
            text = _strip_json_fences(text)
        return LLMResult(text=text, raw=response)
    except groq.AuthenticationError as exc:
        raise ToneShiftError(
            "Invalid API key. Please check GROQ_API_KEY in your .env file.",
            "auth",
        ) from exc
    except groq.RateLimitError as exc:
        raise ToneShiftError(
            "Rate limit exceeded. Please wait a moment and try again.",
            "rate_limit",
        ) from exc
    except groq.PermissionDeniedError as exc:
        raise ToneShiftError(
            "API access forbidden. Verify your Groq API key permissions.",
            "auth",
        ) from exc
    except groq.APIConnectionError as exc:
        raise ToneShiftError(
            "Network error. Check your internet connection and try again.",
            "network",
        ) from exc
    except groq.APIStatusError as exc:
        if getattr(exc, "status_code", None) and exc.status_code >= 500:
            raise ToneShiftError(
                "Groq service is temporarily unavailable. Please try again shortly.",
                "server",
            ) from exc
        raise ToneShiftError(
            f"API request failed: {exc}",
            "api",
        ) from exc
    except groq.APIError as exc:
        raise ToneShiftError(
            f"Groq API error: {exc}",
            "api",
        ) from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise ToneShiftError(
            "Network error. Check your internet connection and try again.",
            "network",
        ) from exc
    except Exception as exc:
        raise ToneShiftError(
            f"Unexpected error while contacting Groq: {exc}",
            "general",
        ) from exc


def rewrite_text(
    client: Groq,
    *,
    text: str,
    tone: str,
    audience: str,
    length: int,
    formality: int,
    creativity: int,
    preserve_technical: bool,
    keep_formatting: bool,
    maintain_bullets: bool,
    keep_numbers: bool,
) -> LLMResult:
    if not text or not text.strip():
        raise ToneShiftError("Please enter some text to rewrite.", "validation")

    prompt = build_rewrite_user_prompt(
        text=text.strip(),
        tone=tone,
        audience=audience,
        length=length,
        formality=formality,
        creativity=creativity,
        preserve_technical=preserve_technical,
        keep_formatting=keep_formatting,
        maintain_bullets=maintain_bullets,
        keep_numbers=keep_numbers,
    )

    temperature = 0.2 + (creativity / 100) * 0.6
    return _call_groq(
        client,
        system_instruction=REWRITE_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=temperature,
    )


def back_translate(client: Groq, rewritten_text: str) -> LLMResult:
    if not rewritten_text or not rewritten_text.strip():
        raise ToneShiftError("No rewritten text available for meaning check.", "validation")

    return _call_groq(
        client,
        system_instruction=BACK_TRANSLATION_SYSTEM,
        user_prompt=build_back_translation_prompt(rewritten_text.strip()),
        temperature=0.1,
    )


def check_meaning_drift(
    client: Groq,
    *,
    original: str,
    rewritten: str,
) -> MeaningCheckResult:
    neutral = back_translate(client, rewritten)
    result = _call_groq(
        client,
        system_instruction=MEANING_CHECK_SYSTEM,
        user_prompt=build_meaning_check_prompt(original, neutral.text),
        temperature=0.0,
        json_mode=True,
    )

    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ToneShiftError(
            "Could not parse meaning-check response from the model.",
            "invalid_response",
        ) from exc

    return MeaningCheckResult(
        status=str(payload.get("status", "minor_drift")),
        confidence=int(payload.get("confidence", 0)),
        meaning_preservation_score=int(payload.get("meaning_preservation_score", 0)),
        explanation=str(payload.get("explanation", "No explanation provided.")),
        neutral_text=neutral.text,
    )


def evaluate_quality(
    client: Groq,
    *,
    original: str,
    rewritten: str,
    tone: str,
    audience: str,
) -> QualityScores:
    result = _call_groq(
        client,
        system_instruction=QUALITY_SCORE_SYSTEM,
        user_prompt=build_quality_score_prompt(original, rewritten, tone, audience),
        temperature=0.0,
        json_mode=True,
    )

    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ToneShiftError(
            "Could not parse quality score response from the model.",
            "invalid_response",
        ) from exc

    return QualityScores(
        meaning_preservation=int(payload.get("meaning_preservation", 0)),
        grammar=int(payload.get("grammar", 0)),
        readability=int(payload.get("readability", 0)),
        tone_accuracy=int(payload.get("tone_accuracy", 0)),
        audience_match=int(payload.get("audience_match", 0)),
        overall=int(payload.get("overall", 0)),
    )


def status_badge(status: str) -> tuple[str, str]:
    mapping = {
        "meaning_preserved": ("Meaning Preserved ✅", "success"),
        "minor_drift": ("Minor Drift ⚠️", "warning"),
        "major_drift": ("Major Drift ❌", "error"),
    }
    return mapping.get(status, ("Unknown Status", "info"))
