"""Tests for German morphological analyzer."""
import pytest
from german_text_analyzer.morphology import (
    guess_gender, find_verb_prefix, conjugate_regular,
    decompose_compound, analyze_word,
)


class TestGenderPrediction:
    def test_feminine_ung(self):
        assert guess_gender("Bildung") == "die"

    def test_feminine_heit(self):
        assert guess_gender("Freiheit") == "die"

    def test_feminine_schaft(self):
        assert guess_gender("Freundschaft") == "die"

    def test_masculine_ling(self):
        assert guess_gender("Fruehling") == "der"

    def test_neuter_chen(self):
        assert guess_gender("Maedchen") == "das"

    def test_neuter_ment(self):
        assert guess_gender("Dokument") == "das"

    def test_unknown_gender(self):
        assert guess_gender("Stuhl") is None


class TestVerbPrefix:
    def test_separable_prefix(self):
        result = find_verb_prefix("aufmachen")
        assert result["prefix"] == "auf"
        assert result["type"] == "separable"

    def test_inseparable_prefix(self):
        result = find_verb_prefix("bestellen")
        assert result["prefix"] == "be"
        assert result["type"] == "inseparable"

    def test_no_prefix(self):
        result = find_verb_prefix("spielen")
        assert result["type"] == "none"


class TestConjugation:
    def test_regular_verb(self):
        forms = conjugate_regular("spielen")
        assert forms["ich"] == "spiele"
        assert forms["du"] == "spielst"
        assert forms["er/sie/es"] == "spielt"

    def test_stem_ending_t(self):
        forms = conjugate_regular("arbeiten")
        assert forms["du"] == "arbeitest"
        assert forms["er/sie/es"] == "arbeitet"


class TestCompoundDecomposition:
    def test_simple_compound(self):
        parts = decompose_compound("Handschuh")
        assert len(parts) >= 2

    def test_short_word_no_split(self):
        parts = decompose_compound("Haus")
        assert len(parts) == 1


class TestAnalyzeWord:
    def test_noun_analysis(self):
        result = analyze_word("Bildung")
        assert result["is_noun"] is True
        assert result["predicted_gender"] == "die"

    def test_verb_analysis(self):
        result = analyze_word("aufmachen")
        assert result["is_noun"] is False
        assert "verb_prefix" in result
