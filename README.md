# German Text Analyzer

A lightweight NLP pipeline for German text processing. Built for researchers and developers working with German-language data.

## Features

- **Tokenization** -- sentence and word-level tokenization optimized for German
- **POS Tagging** -- part-of-speech tagging using spaCy's German model
- **Named Entity Recognition** -- extract persons, organizations, locations, and more
- **Sentiment Analysis** -- German sentiment classification using transformer models
- **Text Statistics** -- readability scores, word frequency, and corpus statistics
- **Compound Word Splitting** -- decompose German compound nouns

## Installation

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

## Quick Start

```python
from german_text_analyzer import GermanAnalyzer

analyzer = GermanAnalyzer()

text = "Die Universitat Stuttgart ist eine der altesten technischen Hochschulen Deutschlands."

# Full analysis
result = analyzer.analyze(text)
print(result.entities)      # Named entities
print(result.pos_tags)      # POS tags
print(result.sentiment)     # Sentiment score
print(result.statistics)    # Text statistics
```

## CLI Usage

```bash
# Analyze a text file
python -m german_text_analyzer analyze --input text.txt --output results.json

# Interactive mode
python -m german_text_analyzer interactive

# Batch processing
python -m german_text_analyzer batch --input-dir ./texts/ --output-dir ./results/
```

## Architecture

```
german_text_analyzer/
    __init__.py          # Public API
    analyzer.py          # Main pipeline orchestrator
    tokenizer.py         # German-specific tokenization
    pos_tagger.py        # POS tagging wrapper
    ner.py               # Named entity recognition
    sentiment.py         # Sentiment analysis
    statistics.py        # Text statistics and readability
    compounds.py         # Compound word splitting
    utils.py             # Shared utilities
```

## Examples

See the `examples/` directory for Jupyter notebooks and scripts:
- `basic_analysis.py` -- single document analysis
- `batch_processing.py` -- process multiple files
- `custom_pipeline.py` -- build custom analysis pipelines

## Tech Stack

- Python 3.9+
- spaCy (German models)
- Hugging Face Transformers
- NLTK

## License

MIT
