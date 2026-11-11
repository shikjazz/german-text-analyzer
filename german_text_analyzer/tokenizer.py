"""German-specific tokenization with sentence and word-level splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Common German abbreviations that should not trigger sentence splits
GERMAN_ABBREVIATIONS = {
    "dr.", "prof.", "hr.", "fr.", "nr.", "bzw.", "usw.", "etc.",
    "ca.", "z.b.", "d.h.", "u.a.", "v.a.", "s.o.", "s.u.",
    "ggf.", "evtl.", "bzgl.", "inkl.", "exkl.", "max.", "min.",
    "str.", "abs.", "tel.", "jan.", "feb.", "marz.", "apr.",
    "jun.", "jul.", "aug.", "sep.", "okt.", "nov.", "dez.",
}

# Patterns for German-specific token handling
ORDINAL_PATTERN = re.compile(r"^\d+\.$")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")


@dataclass
class Token:
    """A single token with metadata."""

    text: str
    start: int
    end: int
    is_punctuation: bool = False
    is_number: bool = False
    is_url: bool = False
    is_email: bool = False


@dataclass
class Sentence:
    """A sentence containing tokens."""

    text: str
    tokens: list[Token] = field(default_factory=list)
    start: int = 0
    end: int = 0


class GermanTokenizer:
    """Tokenizer optimized for German text.

    Handles German-specific challenges:
    - Compound nouns (Donaudampfschifffahrtsgesellschaft)
    - Abbreviations (z.B., d.h., usw.)
    - Ordinal numbers (1., 2., 3.)
    - Umlauts and eszett
    """

    def __init__(self, abbreviations: set[str] | None = None):
        self.abbreviations = abbreviations or GERMAN_ABBREVIATIONS

    def tokenize(self, text: str) -> list[Sentence]:
        """Tokenize text into sentences and words.

        Args:
            text: Input German text.

        Returns:
            List of Sentence objects, each containing Token objects.
        """
        sentences = self._split_sentences(text)
        result = []

        for sent_text, start, end in sentences:
            tokens = self._split_words(sent_text, offset=start)
            sentence = Sentence(
                text=sent_text,
                tokens=tokens,
                start=start,
                end=end,
            )
            result.append(sentence)

        return result

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences, respecting German abbreviations."""
        sentences = []
        current_start = 0
        i = 0

        while i < len(text):
            if text[i] in ".!?":
                # Check if this is an abbreviation
                word_start = self._find_word_start(text, i)
                word = text[word_start : i + 1].lower()

                is_abbrev = word in self.abbreviations
                is_ordinal = bool(ORDINAL_PATTERN.match(word))

                if not is_abbrev and not is_ordinal:
                    # Check for ellipsis
                    if text[i] == "." and i + 2 < len(text) and text[i + 1 : i + 3] == "..":
                        i += 3
                        continue

                    # End of sentence
                    end = i + 1
                    # Skip trailing whitespace for the sentence boundary
                    sent_text = text[current_start:end].strip()
                    if sent_text:
                        sentences.append((sent_text, current_start, end))

                    # Skip whitespace before next sentence
                    i += 1
                    while i < len(text) and text[i].isspace():
                        i += 1
                    current_start = i
                    continue

            i += 1

        # Handle last sentence
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append((remaining, current_start, len(text)))

        return sentences

    def _split_words(self, text: str, offset: int = 0) -> list[Token]:
        """Split a sentence into word tokens."""
        tokens = []
        i = 0

        while i < len(text):
            # Skip whitespace
            if text[i].isspace():
                i += 1
                continue

            # Check for URLs
            url_match = URL_PATTERN.match(text[i:])
            if url_match:
                token_text = url_match.group()
                tokens.append(Token(
                    text=token_text,
                    start=offset + i,
                    end=offset + i + len(token_text),
                    is_url=True,
                ))
                i += len(token_text)
                continue

            # Check for email
            email_match = EMAIL_PATTERN.match(text[i:])
            if email_match:
                token_text = email_match.group()
                tokens.append(Token(
                    text=token_text,
                    start=offset + i,
                    end=offset + i + len(token_text),
                    is_email=True,
                ))
                i += len(token_text)
                continue

            # Punctuation
            if not text[i].isalnum() and text[i] not in "-'":
                tokens.append(Token(
                    text=text[i],
                    start=offset + i,
                    end=offset + i + 1,
                    is_punctuation=True,
                ))
                i += 1
                continue

            # Regular word or number
            start = i
            while i < len(text) and (text[i].isalnum() or text[i] in "-'"):
                i += 1

            token_text = text[start:i]
            tokens.append(Token(
                text=token_text,
                start=offset + start,
                end=offset + i,
                is_number=token_text.replace(".", "").replace(",", "").isdigit(),
            ))

        return tokens

    @staticmethod
    def _find_word_start(text: str, pos: int) -> int:
        """Find the start of the word ending at pos."""
        i = pos - 1
        while i >= 0 and (text[i].isalnum() or text[i] == "."):
            i -= 1
        return i + 1
