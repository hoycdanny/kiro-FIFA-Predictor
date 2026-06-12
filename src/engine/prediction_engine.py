"""
Prediction Engine - Main controller for the FIFA Predictor Power.

Orchestrates ensemble prediction, applies coach style adjustments,
computes confidence index, over/under 2.5 goals, and top 3 scores.
Also handles group stage simulation with standings computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np

from src.data.data_manager import DataManager, TeamProfile
from src.engine.coach_style import CoachStyleSystem, CoachStyleType, SimplePrediction
from src.engine.dixon_coles import DixonColesModel
from src.engine.ensemble import EnsembleModel


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MatchPrediction:
    """Single match prediction result."""

    team_a: str
    team_b: str
    win_prob: float  # Team A win probability (0-1)
    draw_prob: float  # Draw probability (0-1)
    lose_prob: float  # Team A lose probability (0-1)
    top_scores: list[tuple[int, int, float]]  # [(scoreA, scoreB, probability)], top 3
    confidence_index: int  # 0-100 integer
    over_2_5: float  # Over 2.5 goals probability
    under_2_5: float  # Under 2.5 goals probability
    expected_goals_a: float  # Team A expected goals
    expected_goals_b: float  # Team B expected goals
    coach_style: str  # Style used


@dataclass
class GroupStanding:
    """A team's standing within a group."""

    team: str
    played: int  # always 3
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int
    qualification_status: str  # "確定晉級", "可能晉級 (XX%)", ""


@dataclass
class GroupPrediction:
    """Group stage prediction result."""

    group_id: str
    standings: list[GroupStanding]  # 4 teams, ranked
    match_predictions: list[MatchPrediction]  # 6 matches


# ============================================================================
# PREDICTION ENGINE
# ============================================================================


class PredictionEngine:
    """Main prediction engine orchestrating all sub-models.

    Coordinates the ensemble model, Dixon-Coles model, and coach style system
    to produce match and group predictions.
    """

    # Third place qualification probability baseline
    # Based on 2026 WC format: 8 best third-place teams advance from 12 groups
    THIRD_PLACE_BASE_PROBABILITY: float = 0.67  # 8/12

    def __init__(
        self,
        data_manager: DataManager,
        ensemble: EnsembleModel,
        coach_style_system: Optional[CoachStyleSystem] = None,
        dixon_coles: Optional[DixonColesModel] = None,
    ):
        """Initialize the prediction engine.

        Args:
            data_manager: Data manager for loading team and group data.
            ensemble: Ensemble model for combined predictions.
            coach_style_system: Coach style system (defaults to new instance).
            dixon_coles: Dixon-Coles model for score matrix (defaults to ensemble's).
        """
        self.data_manager = data_manager
        self.ensemble = ensemble
        self.coach_style = coach_style_system or CoachStyleSystem()
        self.dixon_coles = dixon_coles or ensemble.dixon_coles

        # Cache teams by name for quick lookup
        self._teams_cache: dict[str, TeamProfile] = {}

    def _get_team(self, team_name: str) -> TeamProfile:
        """Get team profile by name, using cache.

        Args:
            team_name: Canonical English team name.

        Returns:
            TeamProfile for the team.

        Raises:
            ValueError: If team not found.
        """
        if not self._teams_cache:
            teams = self.data_manager.load_teams()
            self._teams_cache = {t.name: t for t in teams}

        if team_name not in self._teams_cache:
            raise ValueError(f"Team '{team_name}' not found in database.")

        return self._teams_cache[team_name]

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        coach_style: Optional[str] = None,
    ) -> MatchPrediction:
        """Execute single match prediction.

        Orchestrates:
        1. Get base W/D/L from ensemble model
        2. Get 5x5 score matrix from Dixon-Coles model
        3. Compute top 3 scores from matrix
        4. Compute over/under 2.5 from matrix
        5. Compute expected goals from matrix
        6. Compute confidence index from W/D/L concentration
        7. Apply coach style if specified

        Args:
            team_a: Canonical name of team A.
            team_b: Canonical name of team B.
            coach_style: Optional coach style name (Chinese or English).

        Returns:
            MatchPrediction with all computed fields.
        """
        # Get team profiles
        profile_a = self._get_team(team_a)
        profile_b = self._get_team(team_b)

        # Step 1: Get base (win_a, draw, win_b) from ensemble
        win_a, draw, win_b = self.ensemble.predict(profile_a, profile_b)

        # Step 2: Get 5x5 score matrix from Dixon-Coles
        score_matrix = self.dixon_coles.predict(profile_a, profile_b)

        # Step 3: Get top 3 most likely scores
        top_scores = self._compute_top_scores(score_matrix, n=3)

        # Step 4: Compute over/under 2.5 goals
        over_2_5, under_2_5 = self._compute_over_under(score_matrix)

        # Step 5: Compute expected goals for each team
        expected_goals_a, expected_goals_b = self._compute_expected_goals(score_matrix)

        # Step 6: Compute confidence index
        confidence_index = self._compute_confidence_index(win_a, draw, win_b)

        # Determine coach style to use
        style_type = self._resolve_coach_style(coach_style)
        style_name = style_type.value

        # Step 7: Apply coach style adjustments
        if style_type != CoachStyleType.ANALYST:
            # Build SimplePrediction for coach style system
            simple_pred = SimplePrediction(
                team_a=team_a,
                team_b=team_b,
                win_prob=win_a,
                draw_prob=draw,
                lose_prob=win_b,
                top_scores=top_scores,
                confidence_index=confidence_index,
                over_2_5=over_2_5,
                under_2_5=under_2_5,
                expected_goals_a=expected_goals_a,
                expected_goals_b=expected_goals_b,
                coach_style=style_name,
                team_a_win_streak=profile_a.current_win_streak,
                team_a_loss_streak=profile_a.current_loss_streak,
                team_b_win_streak=profile_b.current_win_streak,
                team_b_loss_streak=profile_b.current_loss_streak,
                team_a_days_rest=self._get_days_rest(profile_a),
                team_b_days_rest=self._get_days_rest(profile_b),
                team_a_revenge=(profile_a.eliminated_by_2022 == team_b),
                team_b_revenge=(profile_b.eliminated_by_2022 == team_a),
            )
            adjusted = self.coach_style.apply_style(simple_pred, style_type)
            win_a = adjusted.win_prob
            draw = adjusted.draw_prob
            win_b = adjusted.lose_prob
            if adjusted.top_scores is not None:
                top_scores = adjusted.top_scores

        # Ensure W+D+L sums to exactly 1.0
        win_a, draw, win_b = self._normalize_wdl(win_a, draw, win_b)

        # Ensure over+under sums to exactly 1.0
        over_2_5, under_2_5 = self._normalize_pair(over_2_5, under_2_5)

        return MatchPrediction(
            team_a=team_a,
            team_b=team_b,
            win_prob=win_a,
            draw_prob=draw,
            lose_prob=win_b,
            top_scores=top_scores,
            confidence_index=confidence_index,
            over_2_5=over_2_5,
            under_2_5=under_2_5,
            expected_goals_a=expected_goals_a,
            expected_goals_b=expected_goals_b,
            coach_style=style_name,
        )

    def predict_group(self, group_id: str) -> GroupPrediction:
        """Predict group standings by simulating all 6 matches.

        Steps:
        1. Get group teams from DataManager
        2. Generate all 6 matches (round-robin of 4 teams)
        3. Predict each match using predict_match
        4. Simulate outcomes: use most likely result for each match
        5. Calculate standings: points (3W+1D), GD, GF
        6. Rank by: points > GD > GF
        7. Top 2: "確定晉級", 3rd: compute qualification probability

        Args:
            group_id: Group identifier (A-L).

        Returns:
            GroupPrediction with standings and match predictions.

        Raises:
            ValueError: If group_id is invalid.
        """
        # Load group teams
        groups = self.data_manager.load_groups()
        group_id_upper = group_id.upper()

        if group_id_upper not in groups:
            raise ValueError(
                f"Invalid group '{group_id}'. Valid groups: A-L."
            )

        team_names = groups[group_id_upper]

        # Generate all 6 round-robin matches
        match_pairs = list(combinations(team_names, 2))

        # Predict each match
        match_predictions: list[MatchPrediction] = []
        for team_a_name, team_b_name in match_pairs:
            prediction = self.predict_match(team_a_name, team_b_name)
            match_predictions.append(prediction)

        # Simulate outcomes and build standings
        standings_data: dict[str, dict] = {
            name: {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
            }
            for name in team_names
        }

        for pred in match_predictions:
            # Determine most likely outcome
            goals_a, goals_b = self._determine_match_outcome(pred)

            # Update standings for team A
            standings_data[pred.team_a]["goals_for"] += goals_a
            standings_data[pred.team_a]["goals_against"] += goals_b

            # Update standings for team B
            standings_data[pred.team_b]["goals_for"] += goals_b
            standings_data[pred.team_b]["goals_against"] += goals_a

            # Determine result
            if goals_a > goals_b:
                standings_data[pred.team_a]["wins"] += 1
                standings_data[pred.team_b]["losses"] += 1
            elif goals_a < goals_b:
                standings_data[pred.team_b]["wins"] += 1
                standings_data[pred.team_a]["losses"] += 1
            else:
                standings_data[pred.team_a]["draws"] += 1
                standings_data[pred.team_b]["draws"] += 1

        # Build GroupStanding objects
        standings: list[GroupStanding] = []
        for team_name, data in standings_data.items():
            gd = data["goals_for"] - data["goals_against"]
            points = 3 * data["wins"] + data["draws"]
            standings.append(
                GroupStanding(
                    team=team_name,
                    played=3,
                    wins=data["wins"],
                    draws=data["draws"],
                    losses=data["losses"],
                    goals_for=data["goals_for"],
                    goals_against=data["goals_against"],
                    goal_difference=gd,
                    points=points,
                    qualification_status="",  # Set below
                )
            )

        # Sort by: points (desc) > goal difference (desc) > goals for (desc)
        standings.sort(
            key=lambda s: (s.points, s.goal_difference, s.goals_for),
            reverse=True,
        )

        # Set qualification status
        if len(standings) >= 2:
            standings[0].qualification_status = "確定晉級"
            standings[1].qualification_status = "確定晉級"

        if len(standings) >= 3:
            # Compute 3rd place qualification probability
            third_prob = self._compute_third_place_probability(standings[2])
            if third_prob >= 50.0:
                standings[2].qualification_status = (
                    f"可能晉級 ({third_prob:.0f}%)"
                )
            else:
                standings[2].qualification_status = (
                    f"可能晉級 ({third_prob:.0f}%)"
                )

        return GroupPrediction(
            group_id=group_id_upper,
            standings=standings,
            match_predictions=match_predictions,
        )

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _compute_top_scores(
        self, matrix: np.ndarray, n: int = 3
    ) -> list[tuple[int, int, float]]:
        """Extract top N most probable scores from the 5x5 matrix.

        Args:
            matrix: 5x5 score probability matrix.
            n: Number of top scores to return.

        Returns:
            List of (goals_a, goals_b, probability) tuples, sorted by
            probability descending.
        """
        scores: list[tuple[int, int, float]] = []
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                scores.append((i, j, float(matrix[i, j])))

        # Sort by probability descending
        scores.sort(key=lambda x: x[2], reverse=True)

        return scores[:n]

    def _compute_over_under(
        self, matrix: np.ndarray
    ) -> tuple[float, float]:
        """Compute over/under 2.5 goals probabilities from score matrix.

        Over 2.5: total goals >= 3 (i.e., i + j >= 3)
        Under 2.5: total goals <= 2 (i.e., i + j <= 2)

        Args:
            matrix: 5x5 score probability matrix.

        Returns:
            Tuple of (over_2_5, under_2_5) probabilities summing to ~1.0.
        """
        over = 0.0
        under = 0.0
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                if i + j >= 3:
                    over += matrix[i, j]
                else:
                    under += matrix[i, j]

        # Normalize to ensure they sum to 1.0
        total = over + under
        if total > 0:
            over /= total
            under /= total
        else:
            over = 0.5
            under = 0.5

        return (over, under)

    def _compute_expected_goals(
        self, matrix: np.ndarray
    ) -> tuple[float, float]:
        """Compute expected goals for each team from score matrix.

        expected_goals_a = sum(i * P(row_i)) for all cells
        expected_goals_b = sum(j * P(col_j)) for all cells

        Args:
            matrix: 5x5 score probability matrix.

        Returns:
            Tuple of (expected_goals_a, expected_goals_b).
        """
        rows, cols = matrix.shape
        total_prob = matrix.sum()

        if total_prob == 0:
            return (0.0, 0.0)

        # Normalize matrix for expected value calculation
        norm_matrix = matrix / total_prob

        # Expected goals for team A: sum over all cells of i * prob[i,j]
        xg_a = 0.0
        for i in range(rows):
            xg_a += i * norm_matrix[i, :].sum()

        # Expected goals for team B: sum over all cells of j * prob[i,j]
        xg_b = 0.0
        for j in range(cols):
            xg_b += j * norm_matrix[:, j].sum()

        return (float(xg_a), float(xg_b))

    def _compute_confidence_index(
        self, win_a: float, draw: float, win_b: float
    ) -> int:
        """Compute confidence index based on probability concentration.

        Higher concentration of one outcome = higher confidence.
        Scale: max(W, D, L) normalized to 0-100 range.

        The minimum max probability is 1/3 (equal split) -> confidence 0
        The maximum max probability is 1.0 (certainty) -> confidence 100

        Formula: confidence = (max_prob - 1/3) / (1.0 - 1/3) * 100

        Args:
            win_a: Team A win probability.
            draw: Draw probability.
            win_b: Team A lose probability.

        Returns:
            Integer confidence index in [0, 100].
        """
        max_prob = max(win_a, draw, win_b)

        # Scale from [1/3, 1.0] to [0, 100]
        min_max = 1.0 / 3.0
        if max_prob <= min_max:
            return 0

        confidence = (max_prob - min_max) / (1.0 - min_max) * 100.0

        # Clamp to [0, 100] and convert to integer
        return max(0, min(100, int(round(confidence))))

    def _resolve_coach_style(self, style: Optional[str]) -> CoachStyleType:
        """Resolve coach style string to CoachStyleType enum.

        Supports Chinese names, English keywords, and direct enum values.
        Defaults to ANALYST if not specified or unrecognized.

        Args:
            style: Style string (e.g., "分析師", "aggressive", "tactician").

        Returns:
            Resolved CoachStyleType.
        """
        if style is None:
            return CoachStyleType.ANALYST

        # Try direct enum value match
        for style_type in CoachStyleType:
            if style == style_type.value:
                return style_type

        # Try keyword mapping
        from src.engine.coach_style import STYLE_KEYWORDS

        style_lower = style.lower().strip()
        if style_lower in STYLE_KEYWORDS:
            return STYLE_KEYWORDS[style_lower]

        # Try partial match on enum names
        style_upper = style.upper().strip()
        for style_type in CoachStyleType:
            if style_upper == style_type.name:
                return style_type

        # Default to analyst
        return CoachStyleType.ANALYST

    def _get_days_rest(self, team: TeamProfile) -> int:
        """Get days since last match for a team.

        Args:
            team: Team profile.

        Returns:
            Days since last match, or 7 (default rest) if unknown.
        """
        if team.last_match_date is None:
            return 7

        from datetime import date, datetime

        try:
            last_match = datetime.fromisoformat(team.last_match_date).date()
            return (date.today() - last_match).days
        except (ValueError, TypeError):
            return 7

    def _determine_match_outcome(
        self, prediction: MatchPrediction
    ) -> tuple[int, int]:
        """Determine the most likely match outcome from a prediction.

        Uses the top score as the predicted outcome. If no top scores
        available, infers from W/D/L probabilities.

        Args:
            prediction: Match prediction with top_scores.

        Returns:
            Tuple of (goals_a, goals_b) representing the most likely outcome.
        """
        if prediction.top_scores:
            # Use the most likely score
            return (prediction.top_scores[0][0], prediction.top_scores[0][1])

        # Fallback: infer from probabilities
        if prediction.win_prob > prediction.draw_prob and prediction.win_prob > prediction.lose_prob:
            return (1, 0)
        elif prediction.lose_prob > prediction.win_prob and prediction.lose_prob > prediction.draw_prob:
            return (0, 1)
        else:
            return (1, 1)

    def _compute_third_place_probability(self, standing: GroupStanding) -> float:
        """Compute 3rd place team's qualification probability.

        In 2026 World Cup, 8 of 12 third-place teams advance.
        Probability is based on points relative to typical threshold.

        Heuristic:
        - 4+ points: high probability (~85%)
        - 3 points: moderate probability (~60%)
        - 2 points: lower probability (~40%)
        - 1 point: low probability (~20%)
        - 0 points: very low (~5%)

        Args:
            standing: The third-place team's standing.

        Returns:
            Qualification probability as percentage (0-100).
        """
        points = standing.points
        gd = standing.goal_difference

        if points >= 6:
            base_prob = 95.0
        elif points >= 4:
            base_prob = 85.0
        elif points == 3:
            base_prob = 60.0
        elif points == 2:
            base_prob = 40.0
        elif points == 1:
            base_prob = 20.0
        else:
            base_prob = 5.0

        # Adjust slightly by goal difference
        gd_adjustment = min(max(gd * 2.0, -10.0), 10.0)
        final_prob = max(0.0, min(100.0, base_prob + gd_adjustment))

        return final_prob

    @staticmethod
    def _normalize_wdl(
        win: float, draw: float, lose: float
    ) -> tuple[float, float, float]:
        """Normalize win/draw/lose probabilities to sum to exactly 1.0.

        Args:
            win: Win probability.
            draw: Draw probability.
            lose: Lose probability.

        Returns:
            Normalized (win, draw, lose) tuple summing to 1.0.
        """
        total = win + draw + lose
        if total <= 0:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (win / total, draw / total, lose / total)

    @staticmethod
    def _normalize_pair(a: float, b: float) -> tuple[float, float]:
        """Normalize a pair of probabilities to sum to exactly 1.0.

        Args:
            a: First probability.
            b: Second probability.

        Returns:
            Normalized (a, b) tuple summing to 1.0.
        """
        total = a + b
        if total <= 0:
            return (0.5, 0.5)
        return (a / total, b / total)
