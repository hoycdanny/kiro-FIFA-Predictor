"""Historical head-to-head prediction model.

Uses historical matchup records between teams to compute win/draw/loss probabilities.
Falls back to a neutral 33/34/33 split when no H2H data is available.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.data_manager import TeamProfile


class H2HModel:
    """Historical head-to-head prediction model.

    Computes win/draw/loss probabilities based on historical matchups between
    two teams. When no historical data exists for a given pair, returns a
    neutral probability split.
    """

    # Default neutral split when no H2H data available
    NEUTRAL_WIN: float = 0.33
    NEUTRAL_DRAW: float = 0.34
    NEUTRAL_LOSE: float = 0.33

    def __init__(self, h2h_data: Optional[dict[tuple[str, str], dict[str, int]]] = None):
        """Initialize with optional H2H records.

        Args:
            h2h_data: Dictionary mapping team pairs to their head-to-head record.
                Format: {("TeamA", "TeamB"): {"wins_a": int, "draws": int, "wins_b": int}}
                The key order matters: ("TeamA", "TeamB") means wins_a counts
                TeamA's wins against TeamB.
        """
        self._h2h_data: dict[tuple[str, str], dict[str, int]] = h2h_data or {}

    def predict(self, team_a: "TeamProfile", team_b: "TeamProfile") -> tuple[float, float, float]:
        """Predict based on historical H2H records.

        Looks up the head-to-head record between team_a and team_b. If a record
        exists, computes probabilities from the historical W/D/L counts. If no
        record exists (in either key order), falls back to the neutral 33/34/33
        split.

        Args:
            team_a: Profile of the first team.
            team_b: Profile of the second team.

        Returns:
            Tuple of (win_a_prob, draw_prob, win_b_prob), always summing to 1.0.
        """
        record = self._lookup_record(team_a.name, team_b.name)

        if record is None:
            return (self.NEUTRAL_WIN, self.NEUTRAL_DRAW, self.NEUTRAL_LOSE)

        wins_a = record["wins_a"]
        draws = record["draws"]
        wins_b = record["wins_b"]
        total = wins_a + draws + wins_b

        if total == 0:
            return (self.NEUTRAL_WIN, self.NEUTRAL_DRAW, self.NEUTRAL_LOSE)

        win_a_prob = wins_a / total
        draw_prob = draws / total
        win_b_prob = wins_b / total

        return (win_a_prob, draw_prob, win_b_prob)

    def _lookup_record(self, name_a: str, name_b: str) -> Optional[dict[str, int]]:
        """Look up H2H record for two teams, checking both key orderings.

        Args:
            name_a: Canonical name of team A.
            name_b: Canonical name of team B.

        Returns:
            The record dict with wins_a/draws/wins_b from team_a's perspective,
            or None if no record exists.
        """
        # Direct lookup
        key = (name_a, name_b)
        if key in self._h2h_data:
            return self._h2h_data[key]

        # Reverse lookup - swap win counts to maintain team_a perspective
        reverse_key = (name_b, name_a)
        if reverse_key in self._h2h_data:
            record = self._h2h_data[reverse_key]
            return {
                "wins_a": record["wins_b"],
                "draws": record["draws"],
                "wins_b": record["wins_a"],
            }

        return None
