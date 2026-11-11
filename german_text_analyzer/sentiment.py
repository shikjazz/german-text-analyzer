"""Sentiment analysis for German text using transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None  # type: ignore[assignment]


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a text segment."""

    text: str
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float  # Confidence score 0-1
    raw_scores: dict[str, float] | None = None

    @property
    def is_positive(self) -> bool:
        return self.label == "POSITIVE"

    @property
    def is_negative(self) -> bool:
        return self.label == "NEGATIVE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:100] + ("..." if len(self.text) > 100 else ""),
            "label": self.label,
            "score": round(self.score, 4),
            "raw_scores": self.raw_scores,
        }


class SentimentAnalyzer:
    """German sentiment analysis using Hugging Face transformers.

    Default model: oliverguhr/german-sentiment-bert
    Supports POSITIVE, NEGATIVE, NEUTRAL classification.
    """

    DEFAULT_MODEL = "oliverguhr/german-sentiment-bert"

    def __init__(self, model: str | None = None, device: int = -1):
        """Initialize the sentiment analyzer.

        Args:
            model: Hugging Face model identifier. Defaults to a
                   German-specific BERT sentiment model.
            device: Device to run on (-1 for CPU, 0+ for GPU).
        """
        if hf_pipeline is None:
            raise ImportError(
                "transformers is required for sentiment analysis. "
                "Install with: pip install transformers torch"
            )

        model = model or self.DEFAULT_MODEL
        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=model,
            device=device,
            truncation=True,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: German text to analyze.

        Returns:
            SentimentResult with label and confidence score.
        """
        result = self._pipeline(text[:512])[0]

        label = self._normalize_label(result["label"])
        return SentimentResult(
            text=text,
            label=label,
            score=result["score"],
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of German texts to analyze.

        Returns:
            List of SentimentResult objects.
        """
        truncated = [t[:512] for t in texts]
        results = self._pipeline(truncated)

        return [
            SentimentResult(
                text=text,
                label=self._normalize_label(r["label"]),
                score=r["score"],
            )
            for text, r in zip(texts, results)
        ]

    def analyze_sentences(self, text: str) -> list[SentimentResult]:
        """Analyze sentiment per sentence.

        Splits text into sentences and analyzes each independently.
        Useful for finding positive/negative segments in longer texts.

        Args:
            text: German text with multiple sentences.

        Returns:
            List of SentimentResult objects, one per sentence.
        """
        # Simple sentence splitting (use GermanTokenizer for better results)
        import re
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        return self.analyze_batch(sentences)

    def get_overall_sentiment(self, text: str) -> dict[str, Any]:
        """Get overall sentiment with sentence-level breakdown.

        Args:
            text: German text to analyze.

        Returns:
            Dictionary with overall sentiment and per-sentence details.
        """
        sentence_results = self.analyze_sentences(text)

        if not sentence_results:
            return {"overall": "NEUTRAL", "confidence": 0.0, "sentences": []}

        # Calculate weighted average
        positive_score = sum(
            r.score for r in sentence_results if r.is_positive
        )
        negative_score = sum(
            r.score for r in sentence_results if r.is_negative
        )

        total = len(sentence_results)
        if positive_score > negative_score:
            overall = "POSITIVE"
            confidence = positive_score / total
        elif negative_score > positive_score:
            overall = "NEGATIVE"
            confidence = negative_score / total
        else:
            overall = "NEUTRAL"
            confidence = 1.0 / total

        return {
            "overall": overall,
            "confidence": round(confidence, 4),
            "positive_ratio": round(
                sum(1 for r in sentence_results if r.is_positive) / total, 4
            ),
            "negative_ratio": round(
                sum(1 for r in sentence_results if r.is_negative) / total, 4
            ),
            "sentences": [r.to_dict() for r in sentence_results],
        }

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize model output labels to POSITIVE/NEGATIVE/NEUTRAL."""
        label = label.upper()
        mapping = {
            "POSITIVE": "POSITIVE",
            "NEGATIVE": "NEGATIVE",
            "NEUTRAL": "NEUTRAL",
            "POS": "POSITIVE",
            "NEG": "NEGATIVE",
            "NEU": "NEUTRAL",
            "1": "NEGATIVE",
            "2": "NEUTRAL",
            "3": "POSITIVE",
        }
        return mapping.get(label, label)
