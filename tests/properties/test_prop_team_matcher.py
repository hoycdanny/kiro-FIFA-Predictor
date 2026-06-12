"""
Property-based tests for TeamMatcher team name resolution.

**Validates: Requirements 1.5, 1.8, 6.1, 6.3**

Tests:
- Property 4: Team name resolution bidirectional consistency
- Property 5: Invalid team suggestion bounds
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.utils.constants import ALL_TEAMS, TEAM_ALIASES
from src.utils.team_matcher import TeamMatcher


# Shared matcher instance for all tests (immutable after construction)
_matcher = TeamMatcher(ALL_TEAMS)


# Build a set of all known aliases (lowercased) for filtering in Property 5
_all_known_aliases: set[str] = set()
for canonical_name in ALL_TEAMS:
    _all_known_aliases.add(canonical_name.lower())
    for alias in TEAM_ALIASES.get(canonical_name, []):
        _all_known_aliases.add(alias.lower())


class TestProperty4TeamNameBidirectionalConsistency:
    """
    Property 4: Team name resolution bidirectional consistency

    For any team in the 48-team dataset, resolving by its English canonical
    name and by its Chinese name SHALL both map to the same canonical TeamProfile.

    **Validates: Requirements 1.5, 1.8**
    """

    @given(team=st.sampled_from(ALL_TEAMS))
    @settings(max_examples=100)
    def test_canonical_and_chinese_resolve_to_same_team(self, team: str) -> None:
        """Both canonical English name and Chinese alias resolve to the same team."""
        # Resolve by canonical English name
        result_english = _matcher.match(team)
        assert result_english.match_type == "exact", (
            f"Expected exact match for canonical name '{team}', "
            f"got match_type='{result_english.match_type}'"
        )
        assert result_english.team_name == team

        # Get the Chinese alias for this team (first alias is always Chinese)
        aliases = TEAM_ALIASES.get(team, [])
        assume(len(aliases) > 0)  # Skip teams with no aliases

        chinese_alias = aliases[0]  # First alias is the Chinese name

        # Resolve by Chinese alias
        result_chinese = _matcher.match(chinese_alias)
        assert result_chinese.match_type == "exact", (
            f"Expected exact match for Chinese alias '{chinese_alias}' of team '{team}', "
            f"got match_type='{result_chinese.match_type}'"
        )
        assert result_chinese.team_name == team, (
            f"Chinese alias '{chinese_alias}' resolved to '{result_chinese.team_name}' "
            f"but expected '{team}'"
        )


class TestProperty5InvalidTeamSuggestionBounds:
    """
    Property 5: Invalid team suggestion bounds

    For any query string that does not match any of the 48 participating teams,
    the system SHALL return at most 3 similar team name suggestions
    (0 ≤ suggestions ≤ 3).

    **Validates: Requirements 6.1, 6.3**
    """

    @given(query=st.text(min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_non_matching_query_returns_at_most_3_suggestions(self, query: str) -> None:
        """Random strings that don't match any team return 0-3 suggestions."""
        # Filter out strings that are actual team names or aliases
        assume(query.strip().lower() not in _all_known_aliases)
        assume(len(query.strip()) > 0)

        result = _matcher.match(query)

        # If it didn't match exactly or as a single fuzzy match, check suggestions
        if result.match_type == "no_match":
            assert 0 <= len(result.suggestions) <= 3, (
                f"Expected 0-3 suggestions for non-matching query '{query}', "
                f"got {len(result.suggestions)} suggestions: {result.suggestions}"
            )
        elif result.match_type == "multiple":
            # Multiple candidates is also acceptable (fuzzy matched multiple teams)
            # But suggestions on the result should still be bounded
            assert len(result.suggestions) <= 3, (
                f"Expected at most 3 suggestions for query '{query}', "
                f"got {len(result.suggestions)}"
            )

    @given(query=st.text(min_size=1, max_size=50))
    @settings(max_examples=200)
    def test_suggest_similar_returns_at_most_3(self, query: str) -> None:
        """The suggest_similar method always returns at most 3 suggestions."""
        assume(len(query.strip()) > 0)

        suggestions = _matcher.suggest_similar(query)
        assert 0 <= len(suggestions) <= 3, (
            f"Expected 0-3 suggestions from suggest_similar for '{query}', "
            f"got {len(suggestions)}: {suggestions}"
        )
