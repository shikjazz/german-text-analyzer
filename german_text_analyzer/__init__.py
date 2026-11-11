"""German Text Analyzer -- A lightweight NLP pipeline for German text."""

from german_text_analyzer.analyzer import GermanAnalyzer
from german_text_analyzer.tokenizer import GermanTokenizer
from german_text_analyzer.ner import EntityRecognizer
from german_text_analyzer.sentiment import SentimentAnalyzer
from german_text_analyzer.statistics import TextStatistics
from german_text_analyzer.compounds import CompoundSplitter

__version__ = "0.1.0"
__all__ = [
    "GermanAnalyzer",
    "GermanTokenizer",
    "EntityRecognizer",
    "SentimentAnalyzer",
    "TextStatistics",
    "CompoundSplitter",
]
