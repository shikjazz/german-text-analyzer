"""Part-of-speech tagging for German text using spaCy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]


@dataclass
class POSToken:
    """A token with part-of-speech information."""

    text: str
    pos: str       # Universal POS tag
    tag: str       # Fine-grained POS tag
    lemma: str     # Base form
    morph: str     # Morphological features

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pos": self.pos,
            "tag": self.tag,
            "lemma": self.lemma,
            "morph": self.morph,
        }


# Universal POS tag descriptions
POS_DESCRIPTIONS = {
    "ADJ": "Adjective",
    "ADP": "Adposition",
    "ADV": "Adverb",
    "AUX": "Auxiliary verb",
    "CCONJ": "Coordinating conjunction",
    "DET": "Determiner",
    "INTJ": "Interjection",
    "NOUN": "Noun",
    "NUM": "Numeral",
    "PART": "Particle",
    "PRON": "Pronoun",
    "PROPN": "Proper noun",
    "PUNCT": "Punctuation",
    "SCONJ": "Subordinating conjunction",
    "SYM": "Symbol",
    "VERB": "Verb",
    "X": "Other",
}


class POSTagger:
    """Part-of-speech tagger for German using spaCy.

    Provides Universal Dependencies POS tags along with
    fine-grained German-specific tags and morphological features.
    """

    def __init__(self, model: str = "de_core_news_sm"):
        if spacy is None:
            raise ImportError(
                "spaCy is required. Install with: pip install spacy && "
                "python -m spacy download de_core_news_sm"
            )
        self.nlp = spacy.load(model)

    def tag(self, text: str) -> list[POSToken]:
        """Tag all tokens in the text.

        Args:
            text: German text to tag.

        Returns:
            List of POSToken objects.
        """
        doc = self.nlp(text)
        return [
            POSToken(
                text=token.text,
                pos=token.pos_,
                tag=token.tag_,
                lemma=token.lemma_,
                morph=str(token.morph),
            )
            for token in doc
            if not token.is_space
        ]

    def get_nouns(self, text: str) -> list[str]:
        """Extract all nouns from text."""
        tokens = self.tag(text)
        return [t.text for t in tokens if t.pos in ("NOUN", "PROPN")]

    def get_verbs(self, text: str) -> list[str]:
        """Extract all verbs from text."""
        tokens = self.tag(text)
        return [t.text for t in tokens if t.pos in ("VERB", "AUX")]

    def get_adjectives(self, text: str) -> list[str]:
        """Extract all adjectives from text."""
        tokens = self.tag(text)
        return [t.text for t in tokens if t.pos == "ADJ"]

    def lemmatize(self, text: str) -> list[str]:
        """Return lemmatized (base form) tokens."""
        tokens = self.tag(text)
        return [t.lemma for t in tokens if t.pos != "PUNCT"]

    def pos_distribution(self, text: str) -> dict[str, int]:
        """Count the distribution of POS tags in the text."""
        tokens = self.tag(text)
        dist: dict[str, int] = {}
        for t in tokens:
            desc = POS_DESCRIPTIONS.get(t.pos, t.pos)
            dist[desc] = dist.get(desc, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))
