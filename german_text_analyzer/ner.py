"""Named Entity Recognition for German text using spaCy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]


@dataclass
class Entity:
    """A recognized named entity."""

    text: str
    label: str
    start: int
    end: int
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }


# Mapping of spaCy entity labels to human-readable descriptions
ENTITY_DESCRIPTIONS = {
    "PER": "Person",
    "LOC": "Location",
    "ORG": "Organization",
    "MISC": "Miscellaneous",
    "GPE": "Geopolitical Entity",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Monetary Value",
    "PERCENT": "Percentage",
    "QUANTITY": "Quantity",
    "ORDINAL": "Ordinal Number",
    "CARDINAL": "Cardinal Number",
}


class EntityRecognizer:
    """Named Entity Recognition for German text.

    Uses spaCy's German language models for entity extraction.
    Supports custom entity types and post-processing rules.
    """

    def __init__(self, model: str = "de_core_news_sm"):
        """Initialize the NER component.

        Args:
            model: spaCy model name. Options:
                - de_core_news_sm (small, fast)
                - de_core_news_md (medium, balanced)
                - de_core_news_lg (large, accurate)
        """
        if spacy is None:
            raise ImportError(
                "spaCy is required for NER. Install with: pip install spacy && "
                "python -m spacy download de_core_news_sm"
            )
        self.nlp = spacy.load(model)
        self._custom_rules: list[dict[str, Any]] = []

    def recognize(self, text: str) -> list[Entity]:
        """Extract named entities from German text.

        Args:
            text: Input text to analyze.

        Returns:
            List of Entity objects found in the text.
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                description=ENTITY_DESCRIPTIONS.get(ent.label_, "Unknown"),
            )
            entities.append(entity)

        # Apply custom rules
        for rule in self._custom_rules:
            entities.extend(self._apply_rule(text, rule))

        return self._deduplicate(entities)

    def add_rule(
        self,
        pattern: str,
        label: str,
        description: str = "",
    ) -> None:
        """Add a custom entity recognition rule.

        Args:
            pattern: Regex pattern to match.
            label: Entity label to assign.
            description: Human-readable description.
        """
        import re

        self._custom_rules.append({
            "pattern": re.compile(pattern),
            "label": label,
            "description": description,
        })

    def get_entity_types(self, text: str) -> dict[str, list[Entity]]:
        """Group recognized entities by their type.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary mapping entity labels to lists of entities.
        """
        entities = self.recognize(text)
        grouped: dict[str, list[Entity]] = {}

        for entity in entities:
            if entity.label not in grouped:
                grouped[entity.label] = []
            grouped[entity.label].append(entity)

        return grouped

    @staticmethod
    def _apply_rule(text: str, rule: dict[str, Any]) -> list[Entity]:
        """Apply a custom regex rule to find entities."""
        entities = []
        for match in rule["pattern"].finditer(text):
            entities.append(Entity(
                text=match.group(),
                label=rule["label"],
                start=match.start(),
                end=match.end(),
                description=rule.get("description", ""),
            ))
        return entities

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities, keeping the first occurrence."""
        seen: set[tuple[int, int, str]] = set()
        unique = []

        for entity in entities:
            key = (entity.start, entity.end, entity.label)
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return sorted(unique, key=lambda e: e.start)
