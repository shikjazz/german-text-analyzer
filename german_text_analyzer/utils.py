"""Shared utilities for German text processing."""

from __future__ import annotations

import re


def normalize_umlauts(text: str, mode: str = "unicode") -> str:
    """Normalize German umlauts.

    Args:
        text: Input text.
        mode: 'unicode' keeps umlauts, 'ascii' converts ae/oe/ue.

    Returns:
        Normalized text.
    """
    if mode == "ascii":
        replacements = {
            "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
            "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
    return text


def clean_text(text: str) -> str:
    """Clean and normalize German text.

    Removes extra whitespace, normalizes quotes, and fixes
    common encoding issues.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize quotes
    text = text.replace("„", '"').replace("“", '"')
    text = text.replace("‚", "'").replace("‘", "'")
    return text.strip()


def detect_language(text: str) -> dict[str, float]:
    """Simple German language detection heuristic.

    Returns confidence scores for German vs other languages
    based on character and word patterns.
    """
    german_indicators = [
        r"\b(der|die|das|ein|eine|und|oder|nicht|ist|sind)\b",
        r"\b(ich|du|er|sie|es|wir|ihr)\b",
        r"[äöüßÄÖÜ]",
        r"\b(haben|werden|konnen|mussen|sollen)\b",
    ]

    text_lower = text.lower()
    total_words = len(text_lower.split())
    if total_words == 0:
        return {"de": 0.0, "other": 1.0}

    german_matches = 0
    for pattern in german_indicators:
        german_matches += len(re.findall(pattern, text_lower))

    ratio = german_matches / total_words
    confidence = min(ratio * 3, 1.0)  # Scale up, cap at 1.0

    return {"de": round(confidence, 4), "other": round(1 - confidence, 4)}
