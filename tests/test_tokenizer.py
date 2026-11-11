"""Tests for German tokenizer."""

import pytest

from german_text_analyzer.tokenizer import GermanTokenizer


@pytest.fixture
def tokenizer():
    return GermanTokenizer()


class TestSentenceSplitting:
    def test_simple_sentences(self, tokenizer):
        text = "Das ist gut. Das ist schlecht."
        sentences = tokenizer.tokenize(text)
        assert len(sentences) == 2
        assert sentences[0].text == "Das ist gut."
        assert sentences[1].text == "Das ist schlecht."

    def test_abbreviations_not_split(self, tokenizer):
        text = "Dr. Muller arbeitet an der Universitat."
        sentences = tokenizer.tokenize(text)
        assert len(sentences) == 1

    def test_exclamation_and_question(self, tokenizer):
        text = "Was ist das? Das ist toll!"
        sentences = tokenizer.tokenize(text)
        assert len(sentences) == 2

    def test_ordinal_numbers(self, tokenizer):
        text = "Am 1. Januar beginnt das neue Jahr."
        sentences = tokenizer.tokenize(text)
        assert len(sentences) == 1


class TestWordTokenization:
    def test_simple_words(self, tokenizer):
        text = "Hallo Welt"
        sentences = tokenizer.tokenize(text)
        tokens = sentences[0].tokens
        assert len(tokens) == 2
        assert tokens[0].text == "Hallo"
        assert tokens[1].text == "Welt"

    def test_punctuation_separated(self, tokenizer):
        text = "Hallo, Welt!"
        sentences = tokenizer.tokenize(text)
        tokens = sentences[0].tokens
        assert any(t.text == "," and t.is_punctuation for t in tokens)

    def test_urls_preserved(self, tokenizer):
        text = "Besuche https://example.com heute."
        sentences = tokenizer.tokenize(text)
        tokens = sentences[0].tokens
        url_tokens = [t for t in tokens if t.is_url]
        assert len(url_tokens) == 1
        assert "example.com" in url_tokens[0].text

    def test_numbers_detected(self, tokenizer):
        text = "Es sind 42 Studenten."
        sentences = tokenizer.tokenize(text)
        tokens = sentences[0].tokens
        number_tokens = [t for t in tokens if t.is_number]
        assert len(number_tokens) == 1
        assert number_tokens[0].text == "42"
