"""
Property-based tests for the Coach Style System.

**Validates: Requirements 4.4, 8.1, 8.2, 8.7**
"""

from hypothesis import given, settings, assume
from hypothesis.strategies import (
    booleans,
    composite,
    floats,
    integers,
    sampled_from,
    text,
)

from src.engine.coach_style import (
    CoachStyleSystem,
    CoachStyleType,
    SimplePrediction,
    STYLE_NARRATIVE_PREFIX,
)


TEAM_NAMES_A = [
    "Brazil", "Argentina", "France", "Germany", "England",
    "Spain", "Portugal", "Netherlands", "Italy", "Croatia",
]

TEAM_NAMES_B = [
    "Japan", "South Korea", "Australia", "United States", "Mexico",
    "Morocco", "Senegal", "Nigeria", "Canada", "Uruguay",
]


@composite
def simple_predictions(draw):
    """
    Strategy to generate SimplePrediction instances with probabilities summing to 1.0.

    Generates 3 probabilities each in [0.05, 0.90] that sum to 1.0.
    """
    # Generate win in a range that leaves room for draw and lose (each >= 0.05)
    win = draw(floats(min_value=0.05, max_value=0.85))
    # draw must leave at least 0.05 for lose
    max_draw = min(0.85, 1.0 - win - 0.05)
    draw_prob = draw(floats(min_value=0.05, max_value=max_draw))
    lose = 1.0 - win - draw_prob

    # Safety check
    assume(lose >= 0.05)

    team_a = draw(sampled_from(TEAM_NAMES_A))
    team_b = draw(sampled_from(TEAM_NAMES_B))

    # Dynamic context fields
    win_streak_a = draw(integers(min_value=0, max_value=8))
    loss_streak_a = draw(integers(min_value=0, max_value=8))
    win_streak_b = draw(integers(min_value=0, max_value=8))
    loss_streak_b = draw(integers(min_value=0, max_value=8))
    days_rest_a = draw(integers(min_value=1, max_value=10))
    days_rest_b = draw(integers(min_value=1, max_value=10))
    revenge_a = draw(booleans())
    revenge_b = draw(booleans())

    return SimplePrediction(
        team_a=team_a,
        team_b=team_b,
        win_prob=win,
        draw_prob=draw_prob,
        lose_prob=lose,
        top_scores=[(1, 0, 0.3), (2, 1, 0.2), (1, 1, 0.15)],
        confidence_index=draw(integers(min_value=1, max_value=100)),
        over_2_5=0.5,
        under_2_5=0.5,
        expected_goals_a=1.5,
        expected_goals_b=1.0,
        coach_style="分析師",
        team_a_win_streak=win_streak_a,
        team_a_loss_streak=loss_streak_a,
        team_b_win_streak=win_streak_b,
        team_b_loss_streak=loss_streak_b,
        team_a_days_rest=days_rest_a,
        team_b_days_rest=days_rest_b,
        team_a_revenge=revenge_a,
        team_b_revenge=revenge_b,
    )


@composite
def underdog_predictions(draw):
    """
    Strategy to generate predictions where the underdog's win probability is below 35%.
    This forces the contrarian style to trigger a boost.
    """
    # Ensure one side is clearly the underdog (< 35%)
    underdog_win = draw(floats(min_value=0.05, max_value=0.34))
    # The rest is split between draw and favorite
    remaining = 1.0 - underdog_win
    draw_prob = draw(floats(min_value=0.05, max_value=remaining - 0.05))
    favorite_win = remaining - draw_prob
    assume(favorite_win > 0.35)  # Ensure favorite is clearly stronger

    team_a = draw(sampled_from(TEAM_NAMES_A))
    team_b = draw(sampled_from(TEAM_NAMES_B))

    # Randomly decide if team A or team B is the underdog
    a_is_underdog = draw(booleans())

    if a_is_underdog:
        win_prob = underdog_win
        lose_prob = favorite_win
    else:
        win_prob = favorite_win
        lose_prob = underdog_win

    return SimplePrediction(
        team_a=team_a,
        team_b=team_b,
        win_prob=win_prob,
        draw_prob=draw_prob,
        lose_prob=lose_prob,
        top_scores=[(1, 0, 0.3), (2, 1, 0.2), (1, 1, 0.15)],
        confidence_index=50,
        over_2_5=0.5,
        under_2_5=0.5,
        expected_goals_a=1.5,
        expected_goals_b=1.0,
        coach_style="分析師",
        team_a_win_streak=0,
        team_a_loss_streak=0,
        team_b_win_streak=0,
        team_b_loss_streak=0,
        team_a_days_rest=7,
        team_b_days_rest=7,
        team_a_revenge=False,
        team_b_revenge=False,
    )


class TestAnalystStyleIdentity:
    """
    Property 14: Analyst style identity.

    For any MatchPrediction input, applying the analyst coach style SHALL
    return win/draw/lose probabilities identical to the original input
    (no modification).

    **Validates: Requirements 8.1**
    """

    @given(prediction=simple_predictions())
    @settings(max_examples=200)
    def test_analyst_preserves_probabilities(self, prediction: SimplePrediction):
        """Analyst style must not modify win/draw/lose probabilities."""
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.ANALYST)

        assert abs(result.win_prob - prediction.win_prob) < 1e-9, (
            f"Analyst changed win_prob: {prediction.win_prob} -> {result.win_prob}"
        )
        assert abs(result.draw_prob - prediction.draw_prob) < 1e-9, (
            f"Analyst changed draw_prob: {prediction.draw_prob} -> {result.draw_prob}"
        )
        assert abs(result.lose_prob - prediction.lose_prob) < 1e-9, (
            f"Analyst changed lose_prob: {prediction.lose_prob} -> {result.lose_prob}"
        )

    @given(prediction=simple_predictions())
    @settings(max_examples=200)
    def test_analyst_preserves_team_names(self, prediction: SimplePrediction):
        """Analyst style must not change team names."""
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.ANALYST)

        assert result.team_a == prediction.team_a
        assert result.team_b == prediction.team_b

    @given(prediction=simple_predictions())
    @settings(max_examples=100)
    def test_analyst_sets_coach_style_field(self, prediction: SimplePrediction):
        """Analyst style sets the coach_style field to '分析師'."""
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.ANALYST)

        assert result.coach_style == CoachStyleType.ANALYST.value


class TestContrarianStyleUnderdogBoost:
    """
    Property 15: Contrarian style underdog boost with probability preservation.

    For any prediction where the underdog's win probability is below 35%,
    the contrarian style SHALL boost it to [35%, 40%], and the resulting
    win + draw + lose probabilities SHALL still sum to 100.0%.

    **Validates: Requirements 8.2**
    """

    @given(prediction=underdog_predictions())
    @settings(max_examples=200)
    def test_underdog_boosted_to_range(self, prediction: SimplePrediction):
        """
        When underdog win probability < 35%, contrarian boosts it to [35%, 40%].
        """
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.CONTRARIAN)

        # Determine which team is the underdog
        if prediction.win_prob < prediction.lose_prob:
            # Team A is underdog
            underdog_after = result.win_prob
        else:
            # Team B is underdog
            underdog_after = result.lose_prob

        assert 0.35 - 1e-9 <= underdog_after <= 0.40 + 1e-9, (
            f"Underdog probability {underdog_after} not in [0.35, 0.40]. "
            f"Original: win={prediction.win_prob}, lose={prediction.lose_prob}"
        )

    @given(prediction=underdog_predictions())
    @settings(max_examples=200)
    def test_contrarian_probabilities_sum_to_one(self, prediction: SimplePrediction):
        """After contrarian adjustment, W+D+L must still sum to 1.0."""
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.CONTRARIAN)

        total = result.win_prob + result.draw_prob + result.lose_prob
        assert abs(total - 1.0) < 1e-9, (
            f"Contrarian probabilities sum to {total}, expected 1.0. "
            f"win={result.win_prob}, draw={result.draw_prob}, lose={result.lose_prob}"
        )

    @given(prediction=simple_predictions())
    @settings(max_examples=200)
    def test_contrarian_always_preserves_probability_sum(
        self, prediction: SimplePrediction
    ):
        """For any input, contrarian style maintains W+D+L = 1.0."""
        system = CoachStyleSystem()
        result = system.apply_style(prediction, CoachStyleType.CONTRARIAN)

        total = result.win_prob + result.draw_prob + result.lose_prob
        assert abs(total - 1.0) < 1e-9, (
            f"Contrarian probabilities sum to {total}, expected 1.0. "
            f"Input: win={prediction.win_prob}, draw={prediction.draw_prob}, "
            f"lose={prediction.lose_prob}"
        )


class TestCoachStyleNarrativePrefix:
    """
    Property 16: Coach style narrative prefix.

    For any coach style and for any prediction, the generated narrative text
    SHALL begin with the designated prefix:
    - "根據統計分析…" for analyst
    - "從冷門角度…" for contrarian
    - "考量戰術因素…" for tactician

    **Validates: Requirements 8.7**
    """

    @given(
        prediction=simple_predictions(),
        style=sampled_from(list(CoachStyleType)),
    )
    @settings(max_examples=200)
    def test_narrative_starts_with_correct_prefix(
        self, prediction: SimplePrediction, style: CoachStyleType
    ):
        """Generated narrative must start with the style's designated prefix."""
        system = CoachStyleSystem()
        narrative = system.generate_narrative(style, prediction)

        expected_prefix = STYLE_NARRATIVE_PREFIX[style]
        assert narrative.startswith(expected_prefix), (
            f"Narrative for style '{style.value}' does not start with "
            f"expected prefix '{expected_prefix}'. "
            f"Got: '{narrative[:50]}...'"
        )

    @given(prediction=simple_predictions())
    @settings(max_examples=100)
    def test_analyst_narrative_prefix(self, prediction: SimplePrediction):
        """Analyst narrative must start with '根據統計分析…'."""
        system = CoachStyleSystem()
        narrative = system.generate_narrative(CoachStyleType.ANALYST, prediction)
        assert narrative.startswith("根據統計分析…")

    @given(prediction=simple_predictions())
    @settings(max_examples=100)
    def test_contrarian_narrative_prefix(self, prediction: SimplePrediction):
        """Contrarian narrative must start with '從冷門角度…'."""
        system = CoachStyleSystem()
        narrative = system.generate_narrative(CoachStyleType.CONTRARIAN, prediction)
        assert narrative.startswith("從冷門角度…")

    @given(prediction=simple_predictions())
    @settings(max_examples=100)
    def test_tactician_narrative_prefix(self, prediction: SimplePrediction):
        """Tactician narrative must start with '考量戰術因素…'."""
        system = CoachStyleSystem()
        narrative = system.generate_narrative(CoachStyleType.TACTICIAN, prediction)
        assert narrative.startswith("考量戰術因素…")
