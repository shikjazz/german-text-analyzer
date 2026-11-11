"""German compound word splitting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompoundAnalysis:
    """Result of compound word analysis."""

    word: str
    components: list[str]
    is_compound: bool

    @property
    def head(self) -> str:
        """The head (last component) of a compound."""
        return self.components[-1] if self.components else self.word

    @property
    def modifiers(self) -> list[str]:
        """The modifying components (all but the last)."""
        return self.components[:-1] if len(self.components) > 1 else []


# Common German linking elements in compounds (Fugenelemente)
LINKING_ELEMENTS = {"s", "n", "en", "er", "e", "es", "ns", "ens"}

# Common German word stems for splitting heuristic
# In production, use a dictionary lookup; this is a demonstration set
COMMON_STEMS = {
    "arbeit", "auto", "bahn", "bank", "bau", "berg", "bild", "blatt",
    "buch", "bus", "dampf", "dienst", "donau", "dorf", "eis", "fahrt",
    "feld", "feuer", "film", "flug", "frau", "garten", "gast", "geld",
    "haus", "herr", "hof", "hotel", "hund", "jahr", "kinder", "kirche",
    "kraft", "krieg", "kunde", "kunst", "land", "leben", "lehrer",
    "licht", "luft", "mann", "markt", "meer", "mensch", "milch",
    "musik", "nacht", "platz", "post", "rat", "raum", "recht",
    "reise", "ring", "schiff", "schule", "see", "sonne", "spiel",
    "sport", "staat", "stadt", "stein", "stelle", "strasse", "tag",
    "teil", "tier", "tor", "turm", "uhr", "volk", "wahl", "wald",
    "wasser", "weg", "welt", "werk", "wort", "zeit", "zeitung",
    "zimmer", "zug", "gesellschaft", "fahrt", "verwaltung",
}


class CompoundSplitter:
    """Split German compound nouns into their components.

    Uses a dictionary-based approach with linking element awareness.
    This is a heuristic approach; for production use, consider
    integrating with CharSplit or a dedicated compound splitter.

    Example:
        >>> splitter = CompoundSplitter()
        >>> result = splitter.split("Handschuh")
        >>> result.components
        ['Hand', 'Schuh']
    """

    def __init__(self, dictionary: set[str] | None = None, min_part_length: int = 3):
        """Initialize the compound splitter.

        Args:
            dictionary: Set of known German word stems. If None, uses
                       a built-in common stems set.
            min_part_length: Minimum length for a compound part.
        """
        self.dictionary = dictionary or COMMON_STEMS
        self.min_part_length = min_part_length

    def split(self, word: str) -> CompoundAnalysis:
        """Attempt to split a compound word into components.

        Args:
            word: German word to analyze.

        Returns:
            CompoundAnalysis with the split result.
        """
        if len(word) < self.min_part_length * 2:
            return CompoundAnalysis(word=word, components=[word], is_compound=False)

        components = self._find_split(word.lower())

        if len(components) > 1:
            # Restore original casing for first letter of each part
            cased = []
            pos = 0
            for comp in components:
                if pos < len(word):
                    cased.append(word[pos].upper() + comp[1:])
                    pos += len(comp)
                    # Skip linking elements
                    while pos < len(word) and word[pos:pos + 1].lower() in {"s", "n", "e"}:
                        remaining = word[pos:].lower()
                        skipped = False
                        for le in sorted(LINKING_ELEMENTS, key=len, reverse=True):
                            if remaining.startswith(le):
                                pos += len(le)
                                skipped = True
                                break
                        if not skipped:
                            break
                else:
                    cased.append(comp.capitalize())

            return CompoundAnalysis(word=word, components=cased, is_compound=True)

        return CompoundAnalysis(word=word, components=[word], is_compound=False)

    def _find_split(self, word: str) -> list[str]:
        """Find the best split for a compound word using dynamic programming."""
        n = len(word)
        # dp[i] = best split for word[:i]
        dp: list[list[str] | None] = [None] * (n + 1)
        dp[0] = []

        for i in range(1, n + 1):
            for j in range(max(0, i - 20), i):  # Max part length 20
                if dp[j] is None:
                    continue

                part = word[j:i]
                if len(part) < self.min_part_length:
                    continue

                if part in self.dictionary or self._is_known_word(part):
                    candidate = dp[j] + [part]
                    if dp[i] is None or len(candidate) < len(dp[i]):
                        dp[i] = candidate

                # Try skipping linking elements
                for le in LINKING_ELEMENTS:
                    if i + len(le) <= n and word[i : i + len(le)] == le:
                        extended_part = word[j:i]
                        if extended_part in self.dictionary or self._is_known_word(extended_part):
                            # Mark that we can continue from after the linking element
                            next_pos = i + len(le)
                            if next_pos <= n and dp[next_pos] is None:
                                dp[next_pos] = dp[j] + [extended_part]  # type: ignore[operator]

        if dp[n] is not None and len(dp[n]) > 1:
            return dp[n]

        return [word]

    def _is_known_word(self, word: str) -> bool:
        """Check if a word is in the dictionary (with basic stemming)."""
        if word in self.dictionary:
            return True
        # Try removing common suffixes
        for suffix in ("en", "er", "e", "n", "s"):
            stem = word.rstrip(suffix) if word.endswith(suffix) else None
            if stem and stem in self.dictionary:
                return True
        return False

    def analyze_text(self, text: str) -> list[CompoundAnalysis]:
        """Find and analyze all potential compound words in text.

        Args:
            text: German text to scan for compounds.

        Returns:
            List of CompoundAnalysis for words identified as compounds.
        """
        import re
        words = re.findall(r"\b[A-Z]\w{5,}\b", text)  # Capitalized words, 6+ chars
        results = []

        for word in words:
            analysis = self.split(word)
            if analysis.is_compound:
                results.append(analysis)

        return results
