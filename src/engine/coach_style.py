"""
Coach Style System for the FIFA Predictor Power.

Provides three analysis perspectives:
- Analyst (分析師): Pure statistical output, no adjustments.
- Contrarian (反向思考者): Boosts underdog, recommends upset scores.
- Tactician (戰術家): Adjusts based on streaks, revenge, fatigue.

Each style generates a narrative with a fixed prefix.
"""

import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


# ============================================================================
# COACH STYLE TYPES
# ============================================================================


class CoachStyleType(Enum):
    """Three coaching analysis perspectives."""

    ANALYST = "分析師"
    CONTRARIAN = "反向思考者"
    TACTICIAN = "戰術家"


# ============================================================================
# KEYWORD MAPPINGS
# ============================================================================

STYLE_KEYWORDS: dict[str, CoachStyleType] = {
    "conservative": CoachStyleType.ANALYST,
    "保守": CoachStyleType.ANALYST,
    "aggressive": CoachStyleType.CONTRARIAN,
    "激進": CoachStyleType.CONTRARIAN,
    "balanced": CoachStyleType.TACTICIAN,
    "平衡": CoachStyleType.TACTICIAN,
}

STYLE_NARRATIVE_PREFIX: dict[CoachStyleType, str] = {
    CoachStyleType.ANALYST: "根據統計分析…",
    CoachStyleType.CONTRARIAN: "從冷門角度…",
    CoachStyleType.TACTICIAN: "考量戰術因素…",
}


# ============================================================================
# MATCH PREDICTION PROTOCOL (lightweight interface)
# ============================================================================


class MatchPredictionLike(Protocol):
    """Protocol for objects that look like a MatchPrediction."""

    win_prob: float
    draw_prob: float
    lose_prob: float
    team_a: str
    team_b: str


@dataclass
class SimplePrediction:
    """
    Simple prediction dataclass for use within the coach style system.

    This serves as a lightweight stand-in until the full MatchPrediction
    from prediction_engine.py is available.
    """

    team_a: str
    team_b: str
    win_prob: float   # Team A win probability (0-1)
    draw_prob: float  # Draw probability (0-1)
    lose_prob: float  # Team A lose probability (0-1)
    top_scores: Optional[list[tuple[int, int, float]]] = None
    confidence_index: int = 50
    over_2_5: float = 0.5
    under_2_5: float = 0.5
    expected_goals_a: float = 1.0
    expected_goals_b: float = 1.0
    coach_style: str = "分析師"
    # Dynamic context (used by tactician)
    team_a_win_streak: int = 0
    team_a_loss_streak: int = 0
    team_b_win_streak: int = 0
    team_b_loss_streak: int = 0
    team_a_days_rest: int = 7
    team_b_days_rest: int = 7
    team_a_revenge: bool = False  # Team A seeking revenge vs Team B
    team_b_revenge: bool = False  # Team B seeking revenge vs Team A


# ============================================================================
# COACH STYLE SYSTEM
# ============================================================================


class CoachStyleSystem:
    """
    Applies coach style adjustments to match predictions.

    Each style modifies probabilities differently:
    - Analyst: Identity (no change)
    - Contrarian: Boosts underdog if win_prob < 35%
    - Tactician: Adjusts based on dynamic factors
    """

    # Contrarian parameters
    CONTRARIAN_THRESHOLD: float = 0.35  # 35%
    CONTRARIAN_BOOST_MIN: float = 0.35  # 35%
    CONTRARIAN_BOOST_MAX: float = 0.40  # 40%

    # Tactician adjustment magnitude
    TACTICIAN_STREAK_ADJUSTMENT: float = 0.03  # ±3%
    TACTICIAN_REVENGE_ADJUSTMENT: float = 0.02  # +2%
    TACTICIAN_FATIGUE_ADJUSTMENT: float = 0.02  # -2%
    TACTICIAN_FATIGUE_DAYS_THRESHOLD: int = 3
    TACTICIAN_STREAK_THRESHOLD: int = 3

    def apply_style(
        self, prediction: SimplePrediction, style: CoachStyleType
    ) -> SimplePrediction:
        """
        Apply coach style adjustments to a prediction.

        Args:
            prediction: The base prediction to adjust.
            style: The coach style to apply.

        Returns:
            A new SimplePrediction with adjusted probabilities.
        """
        if style == CoachStyleType.ANALYST:
            return self._apply_analyst(prediction)
        elif style == CoachStyleType.CONTRARIAN:
            return self._apply_contrarian(prediction)
        elif style == CoachStyleType.TACTICIAN:
            return self._apply_tactician(prediction)
        else:
            # Fallback to analyst for unknown styles
            return self._apply_analyst(prediction)

    def _apply_analyst(self, prediction: SimplePrediction) -> SimplePrediction:
        """
        Analyst: No modification, return original prediction unchanged.

        Requirements 8.1: Direct output of statistical model, no adjustments.
        """
        result = deepcopy(prediction)
        result.coach_style = CoachStyleType.ANALYST.value
        return result

    def _apply_contrarian(self, prediction: SimplePrediction) -> SimplePrediction:
        """
        Contrarian: Boost underdog to 35-40% if below 35%.

        Logic:
        1. Determine which team is the underdog (lower win probability).
        2. If underdog's win prob < 35%: boost to random [35%, 40%].
        3. Reduce the favorite's win prob proportionally.
        4. Normalize W+D+L to sum to 100%.
        5. Recommend upset scores (underdog winning).

        Requirements 8.2: Boost lower-ranked team's predicted win probability.
        """
        result = deepcopy(prediction)
        result.coach_style = CoachStyleType.CONTRARIAN.value

        # Determine underdog: team with lower win probability
        # win_prob = Team A wins, lose_prob = Team B wins
        team_a_win = result.win_prob
        team_b_win = result.lose_prob

        if team_a_win < team_b_win:
            # Team A is underdog
            if team_a_win < self.CONTRARIAN_THRESHOLD:
                boosted = random.uniform(
                    self.CONTRARIAN_BOOST_MIN, self.CONTRARIAN_BOOST_MAX
                )
                boost_amount = boosted - team_a_win
                # Reduce Team B's win prob (lose_prob) proportionally
                # Distribute reduction across draw and favorite win
                remaining = result.draw_prob + result.lose_prob
                if remaining > 0:
                    draw_ratio = result.draw_prob / remaining
                    lose_ratio = result.lose_prob / remaining
                    result.win_prob = boosted
                    result.draw_prob -= boost_amount * draw_ratio
                    result.lose_prob -= boost_amount * lose_ratio
                else:
                    result.win_prob = boosted

                # Recommend upset scores (Team A winning)
                if result.top_scores is not None:
                    result.top_scores = [
                        (1, 0, 0.25),
                        (2, 1, 0.20),
                        (1, 0, 0.15),
                    ]
        else:
            # Team B is underdog
            if team_b_win < self.CONTRARIAN_THRESHOLD:
                boosted = random.uniform(
                    self.CONTRARIAN_BOOST_MIN, self.CONTRARIAN_BOOST_MAX
                )
                boost_amount = boosted - team_b_win
                # Reduce Team A's win prob (win_prob) proportionally
                remaining = result.draw_prob + result.win_prob
                if remaining > 0:
                    draw_ratio = result.draw_prob / remaining
                    win_ratio = result.win_prob / remaining
                    result.lose_prob = boosted
                    result.draw_prob -= boost_amount * draw_ratio
                    result.win_prob -= boost_amount * win_ratio
                else:
                    result.lose_prob = boosted

                # Recommend upset scores (Team B winning)
                if result.top_scores is not None:
                    result.top_scores = [
                        (0, 1, 0.25),
                        (1, 2, 0.20),
                        (0, 1, 0.15),
                    ]

        # Normalize to ensure W+D+L = 1.0
        result = self._normalize_probabilities(result)

        return result

    def _apply_tactician(self, prediction: SimplePrediction) -> SimplePrediction:
        """
        Tactician: Adjust based on streaks, revenge factor, fatigue.

        Logic:
        1. Check team dynamic factors (win/loss streaks, fatigue, revenge).
        2. Apply small adjustments (+/- 2-3%) based on applicable factors.
        3. Normalize W+D+L to sum to 100%.

        Requirements 8.3: Check trends, revenge, fatigue and adjust accordingly.
        """
        result = deepcopy(prediction)
        result.coach_style = CoachStyleType.TACTICIAN.value

        # Calculate adjustment for Team A
        team_a_adj = 0.0
        team_b_adj = 0.0

        # Win streak bonus (+3%)
        if result.team_a_win_streak >= self.TACTICIAN_STREAK_THRESHOLD:
            team_a_adj += self.TACTICIAN_STREAK_ADJUSTMENT

        if result.team_b_win_streak >= self.TACTICIAN_STREAK_THRESHOLD:
            team_b_adj += self.TACTICIAN_STREAK_ADJUSTMENT

        # Loss streak penalty (-3%)
        if result.team_a_loss_streak >= self.TACTICIAN_STREAK_THRESHOLD:
            team_a_adj -= self.TACTICIAN_STREAK_ADJUSTMENT

        if result.team_b_loss_streak >= self.TACTICIAN_STREAK_THRESHOLD:
            team_b_adj -= self.TACTICIAN_STREAK_ADJUSTMENT

        # Fatigue penalty (-2%)
        if result.team_a_days_rest < self.TACTICIAN_FATIGUE_DAYS_THRESHOLD:
            team_a_adj -= self.TACTICIAN_FATIGUE_ADJUSTMENT

        if result.team_b_days_rest < self.TACTICIAN_FATIGUE_DAYS_THRESHOLD:
            team_b_adj -= self.TACTICIAN_FATIGUE_ADJUSTMENT

        # Revenge bonus (+2%)
        if result.team_a_revenge:
            team_a_adj += self.TACTICIAN_REVENGE_ADJUSTMENT

        if result.team_b_revenge:
            team_b_adj += self.TACTICIAN_REVENGE_ADJUSTMENT

        # Apply adjustments
        # Team A adjustment affects win_prob (positive = more likely to win)
        # Team B adjustment affects lose_prob (positive = more likely Team B wins)
        result.win_prob += team_a_adj
        result.lose_prob += team_b_adj

        # Adjust draw slightly to compensate (shrink if both increase, grow if both decrease)
        net_change = team_a_adj + team_b_adj
        result.draw_prob -= net_change

        # Clamp all probabilities to [0, 1] before normalization
        result.win_prob = max(0.0, result.win_prob)
        result.draw_prob = max(0.0, result.draw_prob)
        result.lose_prob = max(0.0, result.lose_prob)

        # Normalize to ensure W+D+L = 1.0
        result = self._normalize_probabilities(result)

        return result

    def generate_narrative(
        self, style: CoachStyleType, prediction: SimplePrediction
    ) -> str:
        """
        Generate narrative text with fixed prefix per style.

        Requirements 8.7: Each style has a designated prefix:
        - Analyst: "根據統計分析…"
        - Contrarian: "從冷門角度…"
        - Tactician: "考量戰術因素…"

        Args:
            style: The coach style.
            prediction: The prediction to describe.

        Returns:
            Narrative string starting with the style's fixed prefix.
        """
        prefix = STYLE_NARRATIVE_PREFIX[style]

        team_a = prediction.team_a
        team_b = prediction.team_b
        win_pct = prediction.win_prob * 100
        draw_pct = prediction.draw_prob * 100
        lose_pct = prediction.lose_prob * 100

        if style == CoachStyleType.ANALYST:
            body = (
                f"{team_a} 對 {team_b}，"
                f"勝率 {win_pct:.1f}%、平手 {draw_pct:.1f}%、負率 {lose_pct:.1f}%。"
            )
        elif style == CoachStyleType.CONTRARIAN:
            # Identify the underdog
            if prediction.win_prob < prediction.lose_prob:
                underdog = team_a
                underdog_pct = win_pct
            else:
                underdog = team_b
                underdog_pct = lose_pct
            body = (
                f"{underdog} 具有爆冷潛力，"
                f"調整後勝率為 {underdog_pct:.1f}%，值得關注冷門比分。"
            )
        elif style == CoachStyleType.TACTICIAN:
            factors: list[str] = []
            if prediction.team_a_win_streak >= self.TACTICIAN_STREAK_THRESHOLD:
                factors.append(f"{team_a} 近期連勝")
            if prediction.team_b_win_streak >= self.TACTICIAN_STREAK_THRESHOLD:
                factors.append(f"{team_b} 近期連勝")
            if prediction.team_a_loss_streak >= self.TACTICIAN_STREAK_THRESHOLD:
                factors.append(f"{team_a} 近期連敗")
            if prediction.team_b_loss_streak >= self.TACTICIAN_STREAK_THRESHOLD:
                factors.append(f"{team_b} 近期連敗")
            if prediction.team_a_revenge:
                factors.append(f"{team_a} 復仇動力")
            if prediction.team_b_revenge:
                factors.append(f"{team_b} 復仇動力")
            if prediction.team_a_days_rest < self.TACTICIAN_FATIGUE_DAYS_THRESHOLD:
                factors.append(f"{team_a} 體能疲勞")
            if prediction.team_b_days_rest < self.TACTICIAN_FATIGUE_DAYS_THRESHOLD:
                factors.append(f"{team_b} 體能疲勞")

            if factors:
                factor_str = "、".join(factors)
                body = (
                    f"{team_a} 對 {team_b}，"
                    f"戰術因素包含{factor_str}，"
                    f"調整後勝率 {win_pct:.1f}%、平手 {draw_pct:.1f}%、負率 {lose_pct:.1f}%。"
                )
            else:
                body = (
                    f"{team_a} 對 {team_b}，"
                    f"無顯著戰術因素影響，"
                    f"勝率 {win_pct:.1f}%、平手 {draw_pct:.1f}%、負率 {lose_pct:.1f}%。"
                )
        else:
            body = f"{team_a} 對 {team_b}。"

        return prefix + body

    @staticmethod
    def _normalize_probabilities(prediction: SimplePrediction) -> SimplePrediction:
        """
        Normalize win/draw/lose probabilities to sum to 1.0.

        Maintains relative proportions if all values are positive.
        """
        total = prediction.win_prob + prediction.draw_prob + prediction.lose_prob

        if total <= 0:
            # Fallback: equal distribution
            prediction.win_prob = 1.0 / 3.0
            prediction.draw_prob = 1.0 / 3.0
            prediction.lose_prob = 1.0 / 3.0
        else:
            prediction.win_prob /= total
            prediction.draw_prob /= total
            prediction.lose_prob /= total

        return prediction
