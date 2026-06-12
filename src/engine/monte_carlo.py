"""Monte Carlo tournament simulator for FIFA World Cup knockout stage.

Simulates the full knockout bracket (round of 32 through final) using
Monte Carlo methods with NumPy vectorization for performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.engine.ensemble import EnsembleModel
    from src.data.data_manager import TeamProfile


@dataclass
class TournamentResult:
    """Single simulation run result."""

    round_of_32: dict[str, str] = field(default_factory=dict)  # match_id -> winner
    round_of_16: dict[str, str] = field(default_factory=dict)
    quarter_finals: dict[str, str] = field(default_factory=dict)
    semi_finals: dict[str, str] = field(default_factory=dict)
    third_place: str = ""
    final_winner: str = ""


@dataclass
class ChampionPrediction:
    """Champion prediction result."""

    predicted_champion: str
    champion_probability: float
    top_5: list[tuple[str, float]]  # [(team_name, championship_probability)]
    round_probabilities: dict[str, dict[str, float]]  # team -> {round: advance_prob}
    confidence_index: int  # 0-100


class MonteCarloSimulator:
    """Monte Carlo simulator for knockout tournament prediction.

    Uses ensemble model predictions and NumPy vectorization to efficiently
    simulate thousands of tournament brackets.

    Args:
        ensemble: The ensemble model used for match predictions.
        teams: Dictionary mapping team name to TeamProfile.
        n_simulations: Number of simulations to run (default 10,000).
    """

    ROUND_NAMES = [
        "round_of_32",
        "round_of_16",
        "quarter_finals",
        "semi_finals",
        "final",
        "champion",
    ]

    def __init__(
        self,
        ensemble: "EnsembleModel",
        teams: dict[str, "TeamProfile"],
        n_simulations: int = 10000,
    ):
        self.ensemble = ensemble
        self.teams = teams
        self.n_simulations = n_simulations
        # Cache match probabilities to avoid recomputation
        self._prob_cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def simulate_tournament(
        self, qualified_teams: list[str], bracket: dict | None = None
    ) -> ChampionPrediction:
        """Run n_simulations of the full knockout bracket.

        Simulates: round of 32, round of 16, quarter-finals,
        semi-finals, third-place match, and final.

        Args:
            qualified_teams: List of 32 team names that qualified for knockout.
            bracket: Optional bracket mapping. If None, teams are paired
                     sequentially (1v2, 3v4, etc.).

        Returns:
            ChampionPrediction with predicted champion, probabilities, and
            per-team round advancement stats.
        """
        if len(qualified_teams) != 32:
            raise ValueError(
                f"Expected 32 qualified teams, got {len(qualified_teams)}"
            )

        # Build bracket matchups: pairs of teams for round of 32
        if bracket is None:
            # Default: pair sequentially
            matchups = [
                (qualified_teams[i], qualified_teams[i + 1])
                for i in range(0, 32, 2)
            ]
        else:
            matchups = self._build_matchups_from_bracket(qualified_teams, bracket)

        # Pre-compute match probabilities for all possible matchups
        self._precompute_probabilities(qualified_teams)

        # Generate all random numbers upfront for vectorized simulation
        # Max matches per simulation: 16 + 8 + 4 + 2 + 1 + 1 = 32
        random_values = np.random.random((self.n_simulations, 32))

        # Track advancement counts per team per round
        # Rounds: round_of_32_win, round_of_16_win, qf_win, sf_win, final_win
        advancement_counts: dict[str, dict[str, int]] = {
            team: {round_name: 0 for round_name in self.ROUND_NAMES}
            for team in qualified_teams
        }

        results: list[TournamentResult] = []

        for sim_idx in range(self.n_simulations):
            result = self._simulate_single_tournament(
                matchups, random_values[sim_idx], advancement_counts
            )
            results.append(result)

        # Calculate probabilities
        round_probabilities = self._calculate_round_probabilities(
            advancement_counts, qualified_teams
        )

        # Determine champion prediction
        champion_counts = {
            team: advancement_counts[team]["champion"]
            for team in qualified_teams
        }
        predicted_champion = max(champion_counts, key=champion_counts.get)  # type: ignore[arg-type]
        champion_probability = champion_counts[predicted_champion] / self.n_simulations

        # Top 5
        sorted_teams = sorted(
            champion_counts.items(), key=lambda x: x[1], reverse=True
        )
        top_5 = [
            (team, count / self.n_simulations)
            for team, count in sorted_teams[:5]
        ]

        # Confidence index
        confidence_index = self._calculate_confidence(results, champion_counts)

        return ChampionPrediction(
            predicted_champion=predicted_champion,
            champion_probability=champion_probability,
            top_5=top_5,
            round_probabilities=round_probabilities,
            confidence_index=confidence_index,
        )

    def _precompute_probabilities(self, teams: list[str]) -> None:
        """Pre-compute match probabilities for all possible team pairs.

        Uses ensemble model to predict match outcomes and caches results
        for efficient lookup during simulation.
        """
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1 :]:
                if (team_a, team_b) not in self._prob_cache:
                    profile_a = self.teams.get(team_a)
                    profile_b = self.teams.get(team_b)

                    if profile_a and profile_b:
                        win_a, draw, win_b = self.ensemble.predict(
                            profile_a, profile_b
                        )
                    else:
                        # Fallback: equal probabilities
                        win_a, draw, win_b = 1 / 3, 1 / 3, 1 / 3

                    self._prob_cache[(team_a, team_b)] = (win_a, draw, win_b)
                    self._prob_cache[(team_b, team_a)] = (win_b, draw, win_a)

    def _simulate_single_tournament(
        self,
        initial_matchups: list[tuple[str, str]],
        random_values: np.ndarray,
        advancement_counts: dict[str, dict[str, int]],
    ) -> TournamentResult:
        """Simulate one full tournament run.

        Args:
            initial_matchups: 16 matchups for round of 32.
            random_values: Pre-generated random numbers for this simulation.
            advancement_counts: Mutable counter to track advancement.

        Returns:
            TournamentResult with all match winners.
        """
        result = TournamentResult()
        rand_idx = 0

        # Round of 32 (16 matches)
        current_winners: list[str] = []
        for i, (team_a, team_b) in enumerate(initial_matchups):
            winner = self._resolve_knockout_match(
                team_a, team_b, random_values[rand_idx]
            )
            rand_idx += 1
            result.round_of_32[f"R32-{i + 1}"] = winner
            advancement_counts[winner]["round_of_32"] += 1
            current_winners.append(winner)

        # Round of 16 (8 matches)
        next_winners: list[str] = []
        for i in range(0, len(current_winners), 2):
            team_a = current_winners[i]
            team_b = current_winners[i + 1]
            winner = self._resolve_knockout_match(
                team_a, team_b, random_values[rand_idx]
            )
            rand_idx += 1
            result.round_of_16[f"R16-{i // 2 + 1}"] = winner
            advancement_counts[winner]["round_of_16"] += 1
            next_winners.append(winner)
        current_winners = next_winners

        # Quarter-finals (4 matches)
        next_winners = []
        for i in range(0, len(current_winners), 2):
            team_a = current_winners[i]
            team_b = current_winners[i + 1]
            winner = self._resolve_knockout_match(
                team_a, team_b, random_values[rand_idx]
            )
            rand_idx += 1
            result.quarter_finals[f"QF-{i // 2 + 1}"] = winner
            advancement_counts[winner]["quarter_finals"] += 1
            next_winners.append(winner)
        current_winners = next_winners

        # Semi-finals (2 matches)
        next_winners = []
        losers: list[str] = []
        for i in range(0, len(current_winners), 2):
            team_a = current_winners[i]
            team_b = current_winners[i + 1]
            winner = self._resolve_knockout_match(
                team_a, team_b, random_values[rand_idx]
            )
            rand_idx += 1
            loser = team_b if winner == team_a else team_a
            result.semi_finals[f"SF-{i // 2 + 1}"] = winner
            advancement_counts[winner]["semi_finals"] += 1
            next_winners.append(winner)
            losers.append(loser)
        current_winners = next_winners

        # Third place match
        third_place_winner = self._resolve_knockout_match(
            losers[0], losers[1], random_values[rand_idx]
        )
        rand_idx += 1
        result.third_place = third_place_winner

        # Final
        finalist_a = current_winners[0]
        finalist_b = current_winners[1]
        final_winner = self._resolve_knockout_match(
            finalist_a, finalist_b, random_values[rand_idx]
        )
        result.final_winner = final_winner
        advancement_counts[final_winner]["final"] += 1
        advancement_counts[final_winner]["champion"] += 1

        return result

    def _resolve_knockout_match(
        self, team_a: str, team_b: str, random_value: float
    ) -> str:
        """Simulate a single knockout match including extra time / penalties.

        In knockout matches, draws must be resolved. If the random value
        falls in the draw range, a penalty shootout is simulated using
        team penalty_shootout_win_rate.

        Args:
            team_a: First team name.
            team_b: Second team name.
            random_value: Pre-generated random number in [0, 1).

        Returns:
            Name of the winning team.
        """
        probs = self._prob_cache.get((team_a, team_b))
        if probs is None:
            # Shouldn't happen after precomputation, but handle gracefully
            probs = (0.4, 0.2, 0.4)

        win_a, draw, win_b = probs

        if random_value < win_a:
            return team_a
        elif random_value < win_a + draw:
            # Draw → resolve via penalty shootout
            return self._resolve_penalty_shootout(team_a, team_b, random_value)
        else:
            return team_b

    def _resolve_penalty_shootout(
        self, team_a: str, team_b: str, random_value: float
    ) -> str:
        """Resolve a drawn knockout match via penalty shootout.

        Uses team penalty_shootout_win_rate to determine outcome.
        If both teams have rates, compares proportionally.
        Falls back to 50/50 if data is unavailable.

        Args:
            team_a: First team name.
            team_b: Second team name.
            random_value: Random value to determine outcome.

        Returns:
            Name of the penalty shootout winner.
        """
        profile_a = self.teams.get(team_a)
        profile_b = self.teams.get(team_b)

        if profile_a and profile_b:
            rate_a = profile_a.penalty_shootout_win_rate
            rate_b = profile_b.penalty_shootout_win_rate

            # Normalize to get relative probability
            total = rate_a + rate_b
            if total > 0:
                prob_a_wins = rate_a / total
            else:
                prob_a_wins = 0.5
        else:
            prob_a_wins = 0.5

        # Use a derived random value for penalty resolution
        # We use the fractional part transformation to get a new "random" value
        penalty_rand = (random_value * 7.919) % 1.0

        if penalty_rand < prob_a_wins:
            return team_a
        else:
            return team_b

    def _calculate_round_probabilities(
        self,
        advancement_counts: dict[str, dict[str, int]],
        qualified_teams: list[str],
    ) -> dict[str, dict[str, float]]:
        """Convert advancement counts to probabilities.

        Args:
            advancement_counts: Raw counts of how often each team advanced.
            qualified_teams: List of all teams in the bracket.

        Returns:
            Dict mapping team -> {round_name: probability}.
        """
        round_probabilities: dict[str, dict[str, float]] = {}
        for team in qualified_teams:
            team_probs: dict[str, float] = {}
            for round_name in self.ROUND_NAMES:
                count = advancement_counts[team][round_name]
                team_probs[round_name] = count / self.n_simulations
            round_probabilities[team] = team_probs
        return round_probabilities

    def _calculate_confidence(
        self,
        results: list[TournamentResult],
        champion_counts: dict[str, int],
    ) -> int:
        """Calculate confidence index based on simulation convergence.

        High confidence (80-100): Top team > 25% AND top 3 teams > 60%
        Medium confidence (50-79): Top team > 15% OR top 3 teams > 45%
        Low confidence (30-49): Results are spread out

        Args:
            results: List of all simulation results.
            champion_counts: Count of championship wins per team.

        Returns:
            Confidence index integer in [0, 100].
        """
        total = self.n_simulations

        sorted_counts = sorted(champion_counts.values(), reverse=True)

        if len(sorted_counts) == 0:
            return 0

        top_1_prob = sorted_counts[0] / total
        top_3_prob = sum(sorted_counts[:3]) / total if len(sorted_counts) >= 3 else top_1_prob

        # High confidence: dominant favorites
        if top_1_prob > 0.25 and top_3_prob > 0.60:
            # Scale between 80-100 based on how dominant
            confidence = int(80 + min(20, (top_1_prob - 0.25) * 100))
        elif top_1_prob > 0.15 or top_3_prob > 0.45:
            # Medium confidence
            confidence = int(50 + min(29, (top_3_prob - 0.45) * 200))
        elif top_1_prob > 0.08:
            # Low-medium
            confidence = int(30 + min(19, (top_1_prob - 0.08) * 250))
        else:
            # Very spread out
            confidence = int(max(10, min(29, top_1_prob * 300)))

        return max(0, min(100, confidence))

    def _build_matchups_from_bracket(
        self, qualified_teams: list[str], bracket: dict
    ) -> list[tuple[str, str]]:
        """Build matchup list from a bracket dictionary.

        Bracket format: {position_index: team_name} or
        {match_id: (team_a_index, team_b_index)}.

        Falls back to sequential pairing if bracket format is unrecognized.

        Args:
            qualified_teams: List of 32 qualified teams.
            bracket: Bracket structure defining matchups.

        Returns:
            List of 16 team pairs for round of 32.
        """
        matchups: list[tuple[str, str]] = []

        # Support format: {"matches": [[idx_a, idx_b], ...]}
        if "matches" in bracket:
            for match in bracket["matches"]:
                idx_a, idx_b = match[0], match[1]
                matchups.append((qualified_teams[idx_a], qualified_teams[idx_b]))
        # Support format: {"0": "TeamA", "1": "TeamB", ...} position-based
        elif all(k.isdigit() for k in bracket.keys()):
            ordered_teams = [
                bracket[str(i)] for i in range(32)
            ]
            matchups = [
                (ordered_teams[i], ordered_teams[i + 1])
                for i in range(0, 32, 2)
            ]
        else:
            # Fallback: sequential pairing
            matchups = [
                (qualified_teams[i], qualified_teams[i + 1])
                for i in range(0, 32, 2)
            ]

        return matchups
