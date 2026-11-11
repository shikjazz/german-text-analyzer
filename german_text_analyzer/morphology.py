"""
German Morphological Analyzer.
Handles German-specific morphological features: declension, conjugation patterns,
and morphological decomposition of complex German word forms.
"""
import re
from typing import Optional


# German noun declension patterns
CASE_SUFFIXES = {
    "nominative": {"der": "", "die": "", "das": "", "die_pl": ""},
    "accusative": {"der": "", "die": "", "das": "", "die_pl": ""},
    "dative": {"der": "", "die": "", "das": "", "die_pl": "n"},
    "genitive": {"der": "s", "die": "", "das": "s", "die_pl": ""},
}

# Common German suffixes and their typical gender
GENDER_SUFFIXES = {
    "ung": "die",      # Bildung, Zeitung
    "heit": "die",     # Freiheit, Schoenheit
    "keit": "die",     # Moeglichkeit
    "schaft": "die",   # Freundschaft, Gesellschaft
    "tion": "die",     # Information, Situation
    "taet": "die",     # Universitaet, Qualitaet
    "ie": "die",       # Energie, Demokratie
    "enz": "die",      # Kompetenz, Frequenz
    "ling": "der",     # Fruehling, Lehrling
    "ismus": "der",    # Kapitalismus, Tourismus
    "or": "der",       # Motor, Humor
    "er": "der",       # Lehrer, Computer
    "chen": "das",     # Maedchen, Haeschen
    "lein": "das",     # Buechlein, Kindlein
    "ment": "das",     # Dokument, Moment
    "um": "das",       # Datum, Museum
    "nis": "das",      # Ergebnis, Erkenntnis
}

# German verb prefixes (separable and inseparable)
SEPARABLE_PREFIXES = [
    "ab", "an", "auf", "aus", "bei", "ein", "fest", "her", "hin",
    "los", "mit", "nach", "vor", "weg", "zu", "zurueck", "zusammen",
]

INSEPARABLE_PREFIXES = [
    "be", "emp", "ent", "er", "ge", "miss", "ver", "zer",
]

# Common German verb conjugation patterns (present tense, regular)
PRESENT_ENDINGS = {
    "ich": "e",
    "du": "st",
    "er/sie/es": "t",
    "wir": "en",
    "ihr": "t",
    "sie/Sie": "en",
}


def guess_gender(noun: str) -> Optional[str]:
    """Guess the grammatical gender of a German noun based on suffix patterns."""
    noun_lower = noun.lower()
    # Check longest suffixes first
    for suffix in sorted(GENDER_SUFFIXES.keys(), key=len, reverse=True):
        if noun_lower.endswith(suffix):
            return GENDER_SUFFIXES[suffix]
    return None


def find_verb_prefix(verb: str) -> dict:
    """Identify if a German verb has a separable or inseparable prefix."""
    verb_lower = verb.lower()
    for prefix in sorted(SEPARABLE_PREFIXES, key=len, reverse=True):
        if verb_lower.startswith(prefix):
            stem = verb_lower[len(prefix):]
            if len(stem) >= 2:
                return {
                    "prefix": prefix,
                    "type": "separable",
                    "stem": stem,
                    "original": verb,
                }
    for prefix in sorted(INSEPARABLE_PREFIXES, key=len, reverse=True):
        if verb_lower.startswith(prefix):
            stem = verb_lower[len(prefix):]
            if len(stem) >= 2:
                return {
                    "prefix": prefix,
                    "type": "inseparable",
                    "stem": stem,
                    "original": verb,
                }
    return {"prefix": None, "type": "none", "stem": verb_lower, "original": verb}


def conjugate_regular(infinitive: str) -> dict:
    """Conjugate a regular German verb in the present tense."""
    stem = infinitive.rstrip("en") if infinitive.endswith("en") else infinitive.rstrip("n")

    # Handle stems ending in -t, -d (need extra 'e')
    needs_e = stem.endswith(("t", "d"))

    conjugations = {}
    for person, ending in PRESENT_ENDINGS.items():
        if needs_e and ending in ("st", "t"):
            conjugations[person] = f"{stem}e{ending}"
        else:
            conjugations[person] = f"{stem}{ending}"

    return conjugations


def decompose_compound(word: str, min_part_len: int = 3) -> list[str]:
    """
    Attempt to decompose a German compound noun.
    Uses a heuristic approach based on common linking elements.
    """
    word_lower = word.lower()
    linking_elements = ["s", "n", "en", "er", "e", "es"]
    candidates = []

    for split_pos in range(min_part_len, len(word_lower) - min_part_len + 1):
        left = word_lower[:split_pos]
        right = word_lower[split_pos:]

        # Direct split
        if len(left) >= min_part_len and len(right) >= min_part_len:
            candidates.append(([left, right], 0))

        # Try removing linking elements
        for link in linking_elements:
            if right.startswith(link) and len(right) > len(link) + min_part_len - 1:
                remainder = right[len(link):]
                if len(remainder) >= min_part_len:
                    candidates.append(([left, remainder], len(link)))

    # Return the split with the most balanced parts
    if candidates:
        candidates.sort(key=lambda x: abs(len(x[0][0]) - len(x[0][1])))
        return candidates[0][0]
    return [word_lower]


def analyze_word(word: str) -> dict:
    """Perform comprehensive morphological analysis of a German word."""
    result = {
        "word": word,
        "lowercase": word.lower(),
        "is_noun": word[0].isupper() if word else False,
        "length": len(word),
    }

    if result["is_noun"]:
        result["predicted_gender"] = guess_gender(word)
        parts = decompose_compound(word)
        if len(parts) > 1:
            result["compound_parts"] = parts
    else:
        prefix_info = find_verb_prefix(word)
        if prefix_info["prefix"]:
            result["verb_prefix"] = prefix_info

    return result


if __name__ == "__main__":
    # Demo: Gender prediction
    test_nouns = [
        "Bildung", "Freiheit", "Freundschaft", "Fruehling",
        "Maedchen", "Dokument", "Ergebnis", "Information",
    ]
    print("=== Gender Prediction ===")
    for noun in test_nouns:
        gender = guess_gender(noun)
        print(f"  {noun}: {gender}")

    # Demo: Verb analysis
    print("\n=== Verb Prefix Analysis ===")
    test_verbs = ["aufmachen", "bestellen", "zuhoeren", "verstehen", "spielen"]
    for verb in test_verbs:
        info = find_verb_prefix(verb)
        print(f"  {verb}: prefix={info['prefix']}, type={info['type']}")

    # Demo: Conjugation
    print("\n=== Regular Verb Conjugation ===")
    conjugated = conjugate_regular("spielen")
    for person, form in conjugated.items():
        print(f"  {person}: {form}")

    # Demo: Compound decomposition
    print("\n=== Compound Decomposition ===")
    compounds = ["Handschuh", "Kindergarten", "Bundesregierung", "Strassenbahn"]
    for word in compounds:
        parts = decompose_compound(word)
        print(f"  {word}: {' + '.join(parts)}")
"""
