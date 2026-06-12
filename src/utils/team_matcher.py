"""
Team name fuzzy matching utility.

Supports matching by English name, Chinese name, abbreviations,
and partial names using difflib.SequenceMatcher.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from src.utils.constants import TEAM_ALIASES, NON_PARTICIPANT_ALIASES


@dataclass
class MatchResult:
    """Result of a team name matching attempt."""

    match_type: str  # "exact", "single", "multiple", "no_match"
    team_name: Optional[str] = None  # canonical name (for exact/single)
    candidates: list[str] = field(default_factory=list)  # for multiple matches
    suggestions: list[str] = field(default_factory=list)  # for no_match (up to 3)


class TeamMatcher:
    """Fuzzy team name matcher supporting English, Chinese, abbreviations."""

    # Threshold for considering a fuzzy match relevant
    FUZZY_THRESHOLD: float = 0.6
    # Threshold for accepting a single fuzzy match automatically
    SINGLE_MATCH_THRESHOLD: float = 0.8

    def __init__(self, team_names: list[str]):
        """
        Initialize the matcher with a list of canonical team names.

        Args:
            team_names: List of canonical English team names.
        """
        self._canonical_names: list[str] = list(team_names)
        self._index: dict[str, str] = {}  # lowercased alias -> canonical name
        self._build_index(team_names)

    def _build_index(self, team_names: list[str]) -> None:
        """
        Build a lookup index mapping all aliases (lowercased) to canonical names.

        The index includes:
        - Canonical English name (lowercased)
        - All aliases from TEAM_ALIASES (Chinese names, abbreviations, alternate spellings)
        - All aliases from NON_PARTICIPANT_ALIASES (for friendly match support)
        """
        for canonical in team_names:
            # Index the canonical name itself
            self._index[canonical.lower()] = canonical

            # Index all aliases for this team (participant teams)
            aliases = TEAM_ALIASES.get(canonical, [])
            for alias in aliases:
                self._index[alias.lower()] = canonical

            # Index all aliases for non-participant teams
            np_aliases = NON_PARTICIPANT_ALIASES.get(canonical, [])
            for alias in np_aliases:
                self._index[alias.lower()] = canonical

    def match(self, query: str) -> MatchResult:
        """
        Fuzzy match a team name query.

        Matching strategy:
        1. Try exact match (case-insensitive) against canonical names and all aliases
        2. If no exact match, use SequenceMatcher to find fuzzy matches with ratio > 0.6
        3. If exactly one fuzzy match with ratio > 0.8, return as "single" match
        4. If multiple fuzzy matches (ratio > 0.6), return as "multiple"
        5. If no match found (ratio < 0.6), suggest up to 3 closest names

        Args:
            query: The team name query string.

        Returns:
            MatchResult with appropriate match_type and data.
        """
        if not query or not query.strip():
            return MatchResult(
                match_type="no_match",
                suggestions=self.suggest_similar(query) if query else [],
            )

        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # Step 1: Exact match (case-insensitive) against index
        if query_lower in self._index:
            canonical = self._index[query_lower]
            return MatchResult(match_type="exact", team_name=canonical)

        # Step 2: Fuzzy matching using SequenceMatcher
        scored_matches: list[tuple[str, float]] = []

        for key, canonical in self._index.items():
            ratio = SequenceMatcher(None, query_lower, key).ratio()
            if ratio > self.FUZZY_THRESHOLD:
                # Avoid duplicates: only keep the highest ratio per canonical name
                existing = next(
                    (i for i, (name, _) in enumerate(scored_matches) if name == canonical),
                    None,
                )
                if existing is not None:
                    if ratio > scored_matches[existing][1]:
                        scored_matches[existing] = (canonical, ratio)
                else:
                    scored_matches.append((canonical, ratio))

        # Sort by ratio descending
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Single high-confidence match
        if len(scored_matches) == 1 and scored_matches[0][1] > self.SINGLE_MATCH_THRESHOLD:
            return MatchResult(match_type="single", team_name=scored_matches[0][0])

        # If multiple matches but one clearly dominates (ratio > 0.8 and next is much lower)
        if (
            len(scored_matches) >= 1
            and scored_matches[0][1] > self.SINGLE_MATCH_THRESHOLD
        ):
            # Check if the top match is significantly better than others
            if len(scored_matches) == 1 or (
                scored_matches[0][1] - scored_matches[1][1] > 0.15
            ):
                return MatchResult(match_type="single", team_name=scored_matches[0][0])

        # Step 4: Multiple fuzzy matches
        if len(scored_matches) > 1:
            candidates = [name for name, _ in scored_matches]
            return MatchResult(match_type="multiple", candidates=candidates)

        # Step 5: No match - provide suggestions
        suggestions = self.suggest_similar(query_stripped)
        return MatchResult(match_type="no_match", suggestions=suggestions)

    def suggest_similar(self, query: str, max_suggestions: int = 3) -> list[str]:
        """
        Return the most similar team names (up to max_suggestions).

        Uses SequenceMatcher ratio against all canonical names and aliases,
        then returns the unique canonical names sorted by best match ratio.

        Args:
            query: The query string to find suggestions for.
            max_suggestions: Maximum number of suggestions to return (default 3).

        Returns:
            List of canonical team names, most similar first.
        """
        if not query or not query.strip():
            return []

        query_lower = query.strip().lower()
        scores: dict[str, float] = {}  # canonical name -> best ratio

        for key, canonical in self._index.items():
            ratio = SequenceMatcher(None, query_lower, key).ratio()
            if canonical not in scores or ratio > scores[canonical]:
                scores[canonical] = ratio

        # Sort by score descending and return top N
        sorted_teams = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_teams[:max_suggestions]]
