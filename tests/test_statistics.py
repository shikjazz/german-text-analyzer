"""Tests for text statistics module."""

import pytest

from german_text_analyzer.statistics import TextStatistics


@pytest.fixture
def stats():
    return TextStatistics()


class TestBasicStats:
    def test_word_count(self, stats):
        result = stats.compute("Eins zwei drei vier funf.")
        assert result.word_count == 5

    def test_sentence_count(self, stats):
        result = stats.compute("Satz eins. Satz zwei. Satz drei.")
        assert result.sentence_count == 3

    def test_empty_text(self, stats):
        result = stats.compute("")
        assert result.word_count == 0

    def test_vocabulary_size(self, stats):
        result = stats.compute("Hallo Welt Hallo Freund")
        assert result.vocabulary_size == 3  # hallo, welt, freund

    def test_type_token_ratio(self, stats):
        result = stats.compute("ja ja ja nein")
        assert result.type_token_ratio == pytest.approx(0.5, abs=0.01)


class TestReadability:
    def test_flesch_in_range(self, stats):
        text = (
            "Die Katze sitzt auf der Matte. "
            "Der Hund spielt im Garten. "
            "Es ist ein sonniger Tag."
        )
        result = stats.compute(text)
        assert 0 <= result.flesch_reading_ease <= 100

    def test_simple_text_higher_score(self, stats):
        simple = "Das ist gut. Das ist schlecht. Das ist so."
        complex_text = (
            "Die Implementierung der Nachhaltigkeitsstrategie "
            "erfordert eine interdisziplinare Zusammenarbeit "
            "verschiedener gesellschaftlicher Institutionen."
        )
        simple_score = stats.compute(simple).flesch_reading_ease
        complex_score = stats.compute(complex_text).flesch_reading_ease
        assert simple_score > complex_score


class TestSyllables:
    def test_single_syllable(self, stats):
        assert stats._count_syllables("Haus") == 1

    def test_multi_syllable(self, stats):
        count = stats._count_syllables("Universitat")
        assert count >= 4

    def test_diphthong(self, stats):
        assert stats._count_syllables("Haus") == 1  # 'au' is a diphthong
