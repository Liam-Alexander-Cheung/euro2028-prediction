"""
Tests for src.name_matching — the string half of cross-source player matching
(normalize_name, name_similarity).

These functions are pure and need no fixtures: the oracle cases are the exact
real-name examples already written into the module's own docstrings. Encoding
them as assertions turns "the docstring claims this" into "CI proves this", and
locks in one documented LIMITATION (the Müller/Mueller gap) so it can't silently
change without a test going red.
"""

from src.name_matching import normalize_name, name_similarity


class TestNormalizeName:
    def test_strips_diacritics(self):
        # NFKD decomposition drops the accents: ü→u, é→e
        assert normalize_name("Müller") == "muller"
        assert normalize_name("Kanté") == "kante"

    def test_turkish_dotted_and_dotless_i(self):
        # İlkay Gündoğan is the canonical Turkish-i / soft-g case
        assert normalize_name("İlkay Gündoğan") == "ilkay gundogan"

    def test_non_decomposing_letters_are_folded(self):
        # ł and ø are distinct letters NFKD leaves alone, so _LETTER_FOLD handles them
        assert normalize_name("Łukasz Fabiański") == "lukasz fabianski"
        assert normalize_name("Ødegaard") == "odegaard"

    def test_captain_marker_is_dropped_without_leaving_a_token(self):
        # "(c)" must vanish entirely, not leave a stray "c" token behind
        assert normalize_name("Lionel Messi (c)") == "lionel messi"

    def test_apostrophe_closes_with_no_gap(self):
        # N'Golo → ngolo (apostrophe removed, NOT turned into a space)
        assert normalize_name("N'Golo Kanté") == "ngolo kante"

    def test_surname_first_is_reordered(self):
        # "Last, First" → "First Last", so it matches the forename-first spelling
        assert normalize_name("Fabiański, Łukasz") == "lukasz fabianski"
        assert normalize_name("Fabiański, Łukasz") == normalize_name("Łukasz Fabiański")

    def test_empty_and_none_return_empty_string(self):
        # callers treat "" as unmatchable — never a wildcard, never a crash
        assert normalize_name("") == ""
        assert normalize_name(None) == ""

    def test_documented_mueller_limitation_is_preserved(self):
        # KNOWN, ACCEPTED gap: the German transliteration "Mueller" does NOT
        # collapse to "muller" (ü→u ≠ ue). This asserts the gap still exists, so
        # if a future change closes it, someone updates the docstring on purpose.
        assert normalize_name("Mueller") == "mueller"
        assert normalize_name("Mueller") != normalize_name("Müller")


class TestNameSimilarity:
    def test_identical_names_score_one(self):
        assert name_similarity("Lionel Messi", "Lionel Messi") == 1.0

    def test_diacritics_do_not_reduce_similarity(self):
        # both sides normalize first, so the accent is gone before comparison
        assert name_similarity("Müller", "Muller") == 1.0

    def test_extra_middle_names_stay_high(self):
        # token_set_ratio is robust to one name carrying extra tokens — exactly
        # what two-surname / middle-name spellings need
        score = name_similarity("Lionel Messi", "Lionel Andres Messi Cuccitini")
        assert score >= 0.9

    def test_empty_name_scores_zero(self):
        # an empty name is never "100% similar" to anything
        assert name_similarity("", "Lionel Messi") == 0.0
        assert name_similarity("Lionel Messi", "") == 0.0

    def test_unrelated_names_score_low(self):
        assert name_similarity("Lionel Messi", "Manuel Neuer") < 0.5
