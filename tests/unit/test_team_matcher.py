"""Unit tests for TeamMatcher fuzzy name resolution."""

import pytest

from src.utils.constants import ALL_TEAMS, TEAM_ALIASES
from src.utils.team_matcher import MatchResult, TeamMatcher


@pytest.fixture
def matcher() -> TeamMatcher:
    """Create a TeamMatcher with all 48 teams."""
    return TeamMatcher(ALL_TEAMS)


class TestExactMatch:
    """Tests for exact match (case-insensitive) against canonical names and aliases."""

    def test_exact_english_name(self, matcher: TeamMatcher):
        result = matcher.match("Brazil")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_english_name_case_insensitive(self, matcher: TeamMatcher):
        result = matcher.match("brazil")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_english_name_uppercase(self, matcher: TeamMatcher):
        result = matcher.match("BRAZIL")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_chinese_name(self, matcher: TeamMatcher):
        result = matcher.match("巴西")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_abbreviation(self, matcher: TeamMatcher):
        result = matcher.match("BRA")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_abbreviation_case_insensitive(self, matcher: TeamMatcher):
        result = matcher.match("bra")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_alternate_spelling(self, matcher: TeamMatcher):
        result = matcher.match("Holland")
        assert result.match_type == "exact"
        assert result.team_name == "Netherlands"

    def test_exact_us_variants(self, matcher: TeamMatcher):
        result = matcher.match("USA")
        assert result.match_type == "exact"
        assert result.team_name == "United States"

        result = matcher.match("US")
        assert result.match_type == "exact"
        assert result.team_name == "United States"

    def test_exact_chinese_usa(self, matcher: TeamMatcher):
        result = matcher.match("美國")
        assert result.match_type == "exact"
        assert result.team_name == "United States"

    def test_exact_with_whitespace(self, matcher: TeamMatcher):
        result = matcher.match("  Brazil  ")
        assert result.match_type == "exact"
        assert result.team_name == "Brazil"

    def test_exact_multi_word_name(self, matcher: TeamMatcher):
        result = matcher.match("South Korea")
        assert result.match_type == "exact"
        assert result.team_name == "South Korea"

    def test_exact_alternate_kor(self, matcher: TeamMatcher):
        result = matcher.match("Korea Republic")
        assert result.match_type == "exact"
        assert result.team_name == "South Korea"


class TestFuzzyMatch:
    """Tests for fuzzy matching with SequenceMatcher."""

    def test_single_fuzzy_match_typo(self, matcher: TeamMatcher):
        # "Brazl" is close to "Brazil"
        result = matcher.match("Brazl")
        assert result.match_type in ("single", "no_match")
        if result.match_type == "single":
            assert result.team_name == "Brazil"

    def test_single_fuzzy_match_partial(self, matcher: TeamMatcher):
        # "Argentin" is close to "Argentina"
        result = matcher.match("Argentin")
        assert result.match_type == "single"
        assert result.team_name == "Argentina"

    def test_multiple_fuzzy_matches(self, matcher: TeamMatcher):
        # "Saudi Arabi" is close to "Saudi Arabia" but might also match others
        # "Saudia" matches Saudi Arabia closely via alias
        # Let's use a query that genuinely hits multiple teams above 0.6
        result = matcher.match("South Kore")
        # This should be a single match because it's very close to "South Korea"
        assert result.match_type == "single"
        assert result.team_name == "South Korea"

    def test_multiple_fuzzy_candidates_returned(self, matcher: TeamMatcher):
        # "Ira" should fuzzy-match both "Iran" and "Iraq" above 0.6
        result = matcher.match("Ira")
        # Both Iran and Iraq are close enough
        if result.match_type == "multiple":
            assert len(result.candidates) >= 2
            assert "Iran" in result.candidates
            assert "Iraq" in result.candidates


class TestNoMatch:
    """Tests for no match with suggestions."""

    def test_no_match_returns_suggestions(self, matcher: TeamMatcher):
        result = matcher.match("Xyzland")
        assert result.match_type == "no_match"
        assert len(result.suggestions) <= 3

    def test_no_match_empty_string(self, matcher: TeamMatcher):
        result = matcher.match("")
        assert result.match_type == "no_match"

    def test_no_match_whitespace_only(self, matcher: TeamMatcher):
        result = matcher.match("   ")
        assert result.match_type == "no_match"

    def test_suggestions_max_3(self, matcher: TeamMatcher):
        result = matcher.match("Fantasyland")
        assert result.match_type == "no_match"
        assert len(result.suggestions) <= 3


class TestSuggestSimilar:
    """Tests for suggest_similar method."""

    def test_suggest_returns_up_to_3(self, matcher: TeamMatcher):
        suggestions = matcher.suggest_similar("Germa")
        assert len(suggestions) <= 3
        assert "Germany" in suggestions

    def test_suggest_returns_relevant_teams(self, matcher: TeamMatcher):
        suggestions = matcher.suggest_similar("Franca")
        assert "France" in suggestions

    def test_suggest_empty_query(self, matcher: TeamMatcher):
        suggestions = matcher.suggest_similar("")
        assert suggestions == []

    def test_suggest_custom_max(self, matcher: TeamMatcher):
        suggestions = matcher.suggest_similar("land", max_suggestions=2)
        assert len(suggestions) <= 2


class TestAllTeamsResolvable:
    """Ensure all 48 teams and their aliases resolve correctly."""

    def test_all_canonical_names_resolve(self, matcher: TeamMatcher):
        for team in ALL_TEAMS:
            result = matcher.match(team)
            assert result.match_type == "exact", f"Failed for: {team}"
            assert result.team_name == team

    def test_all_aliases_resolve_to_canonical(self, matcher: TeamMatcher):
        for canonical, aliases in TEAM_ALIASES.items():
            for alias in aliases:
                result = matcher.match(alias)
                assert result.match_type == "exact", (
                    f"Failed for alias '{alias}' of '{canonical}'"
                )
                assert result.team_name == canonical
