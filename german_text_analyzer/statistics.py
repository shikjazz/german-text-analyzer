"""Text statistics and readability metrics for German text."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class TextStats:
    """Statistical analysis of a German text."""

    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_size: int
    type_token_ratio: float
    flesch_reading_ease: float
    word_frequencies: Counter
    longest_word: str
    longest_sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "vocabulary_size": self.vocabulary_size,
            "type_token_ratio": round(self.type_token_ratio, 4),
            "flesch_reading_ease": round(self.flesch_reading_ease, 2),
            "top_words": self.word_frequencies.most_common(20),
            "longest_word": self.longest_word,
        }


# German stopwords for frequency analysis
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen",
    "einem", "einer", "und", "oder", "aber", "nicht", "ist", "sind",
    "war", "hat", "haben", "wird", "werden", "kann", "mit", "von",
    "zu", "in", "auf", "an", "fur", "um", "aus", "bei", "nach",
    "uber", "vor", "wie", "als", "auch", "noch", "so", "nur",
    "schon", "wenn", "da", "ich", "er", "sie", "es", "wir", "ihr",
    "sich", "man", "dass", "was", "wer", "wo", "diese", "dieser",
    "diesem", "diesen", "jeder", "alle", "kein", "keine", "mehr",
    "sehr", "durch", "zwischen", "unter", "gegen", "ohne", "bis",
}


class TextStatistics:
    """Compute text statistics and readability metrics for German.

    Includes German-adapted Flesch Reading Ease and other metrics
    relevant to German text complexity analysis.
    """

    def __init__(self, stopwords: set[str] | None = None):
        self.stopwords = stopwords or GERMAN_STOPWORDS

    def compute(self, text: str) -> TextStats:
        """Compute all statistics for a text.

        Args:
            text: German text to analyze.

        Returns:
            TextStats object with all computed metrics.
        """
        words = self._extract_words(text)
        sentences = self._extract_sentences(text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        syllable_counts = [self._count_syllables(w) for w in words]

        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        total_syllables = sum(syllable_counts)

        # Word frequencies (excluding stopwords)
        content_words = [
            w.lower() for w in words if w.lower() not in self.stopwords
        ]
        frequencies = Counter(content_words)

        # Type-Token Ratio
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(word_count, 1)

        # German Flesch Reading Ease
        # Adapted formula: 180 - ASL - (58.5 * ASW)
        # ASL = average sentence length, ASW = average syllables per word
        asl = word_count / sentence_count
        asw = total_syllables / max(word_count, 1)
        flesch = 180 - asl - (58.5 * asw)
        flesch = max(0, min(100, flesch))  # Clamp to 0-100

        longest_word = max(words, key=len) if words else ""
        longest_sentence = max(sentences, key=len) if sentences else ""

        return TextStats(
            char_count=len(text),
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=max(len(paragraphs), 1),
            avg_word_length=sum(len(w) for w in words) / max(word_count, 1),
            avg_sentence_length=asl,
            vocabulary_size=len(unique_words),
            type_token_ratio=ttr,
            flesch_reading_ease=flesch,
            word_frequencies=frequencies,
            longest_word=longest_word,
            longest_sentence=longest_sentence,
        )

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a German word.

        Uses vowel-group heuristic adapted for German phonology.
        Handles umlauts and common German diphthongs.
        """
        word = word.lower()
        vowels = set("aeiouyäöü")
        diphthongs = {"ei", "ai", "au", "eu", "äu", "ie"}

        count = 0
        i = 0
        while i < len(word):
            if word[i] in vowels:
                # Check for diphthong
                if i + 1 < len(word) and word[i : i + 2] in diphthongs:
                    count += 1
                    i += 2
                    continue
                count += 1
            i += 1

        return max(count, 1)

    @staticmethod
    def _extract_words(text: str) -> list[str]:
        """Extract words from text."""
        return re.findall(r"\b\w+\b", text)

    @staticmethod
    def _extract_sentences(text: str) -> list[str]:
        """Extract sentences from text."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]
