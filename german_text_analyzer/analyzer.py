"""Main pipeline orchestrator for German text analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from german_text_analyzer.tokenizer import GermanTokenizer, Sentence
from german_text_analyzer.ner import EntityRecognizer, Entity
from german_text_analyzer.sentiment import SentimentAnalyzer, SentimentResult
from german_text_analyzer.statistics import TextStatistics, TextStats
from german_text_analyzer.compounds import CompoundSplitter, CompoundAnalysis


@dataclass
class AnalysisResult:
    """Complete analysis result for a German text."""

    text: str
    sentences: list[Sentence] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    sentiment: SentimentResult | None = None
    statistics: TextStats | None = None
    compounds: list[CompoundAnalysis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text[:200] + ("..." if len(self.text) > 200 else ""),
            "sentence_count": len(self.sentences),
            "entities": [e.to_dict() for e in self.entities],
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "compounds": [
                {"word": c.word, "components": c.components}
                for c in self.compounds
            ],
        }

    def summary(self) -> str:
        """Human-readable summary of the analysis."""
        lines = [f"Analysis of {len(self.text)} characters:"]
        lines.append(f"  Sentences: {len(self.sentences)}")
        lines.append(f"  Entities: {len(self.entities)}")

        if self.sentiment:
            lines.append(
                f"  Sentiment: {self.sentiment.label} "
                f"(confidence: {self.sentiment.score:.2%})"
            )

        if self.statistics:
            lines.append(f"  Words: {self.statistics.word_count}")
            lines.append(f"  Vocabulary: {self.statistics.vocabulary_size}")
            lines.append(
                f"  Readability (Flesch): {self.statistics.flesch_reading_ease:.1f}"
            )

        if self.compounds:
            lines.append(f"  Compound words: {len(self.compounds)}")

        return "\n".join(lines)


class GermanAnalyzer:
    """Full-pipeline German text analyzer.

    Orchestrates tokenization, NER, sentiment analysis,
    text statistics, and compound word splitting.

    Example:
        >>> analyzer = GermanAnalyzer()
        >>> result = analyzer.analyze("Berlin ist die Hauptstadt Deutschlands.")
        >>> print(result.summary())
    """

    def __init__(
        self,
        spacy_model: str = "de_core_news_sm",
        sentiment_model: str | None = None,
        enable_sentiment: bool = True,
        enable_ner: bool = True,
        enable_compounds: bool = True,
    ):
        """Initialize the analyzer pipeline.

        Args:
            spacy_model: spaCy model for NER and POS tagging.
            sentiment_model: Hugging Face model for sentiment.
            enable_sentiment: Whether to run sentiment analysis.
            enable_ner: Whether to run named entity recognition.
            enable_compounds: Whether to run compound splitting.
        """
        self.tokenizer = GermanTokenizer()
        self.statistics = TextStatistics()
        self.compound_splitter = CompoundSplitter()

        self._ner = None
        self._sentiment = None

        if enable_ner:
            try:
                self._ner = EntityRecognizer(model=spacy_model)
            except (ImportError, OSError) as e:
                print(f"Warning: NER disabled ({e})")

        if enable_sentiment:
            try:
                self._sentiment = SentimentAnalyzer(model=sentiment_model)
            except ImportError as e:
                print(f"Warning: Sentiment analysis disabled ({e})")

    def analyze(self, text: str) -> AnalysisResult:
        """Run the full analysis pipeline on German text.

        Args:
            text: German text to analyze.

        Returns:
            AnalysisResult with all analysis components.
        """
        result = AnalysisResult(text=text)

        # Tokenization (always runs)
        result.sentences = self.tokenizer.tokenize(text)

        # Text statistics (always runs)
        result.statistics = self.statistics.compute(text)

        # NER
        if self._ner is not None:
            result.entities = self._ner.recognize(text)

        # Sentiment
        if self._sentiment is not None:
            result.sentiment = self._sentiment.analyze(text)

        # Compound splitting
        result.compounds = self.compound_splitter.analyze_text(text)

        return result

    def analyze_file(self, filepath: str, encoding: str = "utf-8") -> AnalysisResult:
        """Analyze text from a file.

        Args:
            filepath: Path to the text file.
            encoding: File encoding (default: utf-8).

        Returns:
            AnalysisResult for the file content.
        """
        with open(filepath, "r", encoding=encoding) as f:
            text = f.read()
        return self.analyze(text)
