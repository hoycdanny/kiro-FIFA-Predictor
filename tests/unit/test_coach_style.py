"""
Unit tests for the Coach Style System.

Tests:
1. Analyst returns identical probabilities (identity property)
2. Contrarian boosts underdog to [35%, 40%] when below 35%
3. All styles maintain W+D+L = 100%
4. Narrative starts with correct prefix for each style
"""

import pytest
import math

from src.engine.coach_style import (
    CoachStyleSystem,
    CoachStyleType,
    SimplePrediction,
    STYLE_KEYWORDS,
    STYLE_NARRATIVE_PREFIX,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def system() -> CoachStyleSystem:
    """Create a fresh CoachStyleSystem instance."""
    return CoachStyleSystem()


@pytest.fixture
def balanced_prediction() -> SimplePrediction:
    """A balanced prediction where no team is a clear underdog."""
    return SimplePrediction(
        team_a="Brazil",
        team_b="Argentina",
        win_prob=0.40,
        draw_prob=0.25,
        lose_prob=0.35,
    )


@pytest.fixture
def strong_favorite_prediction() -> SimplePrediction:
    """A prediction with a strong favorite (Team A) and underdog (Team B)."""
    return SimplePrediction(
        team_a="Brazil",
        team_b="Jamaica",
        win_prob=0.70,
        draw_prob=0.15,
        lose_prob=0.15,
    )


@pytest.fixture
def weak_team_a_prediction() -> SimplePrediction:
    """A prediction where Team A is the underdog."""
    return SimplePrediction(
        team_a="Jamaica",
        team_b="Brazil",
        win_prob=0.15,
        draw_prob=0.15,
        lose_prob=0.70,
    )


@pytest.fixture
def tactician_prediction() -> SimplePrediction:
    """A prediction with dynamic factors for tactician testing."""
    return SimplePrediction(
        team_a="Argentina",
        team_b="France",
        win_prob=0.45,
        draw_prob=0.25,
        lose_prob=0.30,
        team_a_win_streak=4,
        team_b_loss_streak=0,
        team_a_days_rest=5,
        team_b_days_rest=2,
        team_a_revenge=True,
        team_b_revenge=False,
    )


# ============================================================================
# TEST: ANALYST STYLE (Identity)
# ============================================================================


class TestAnalystStyle:
    """Test that analyst style returns identical probabilities."""

    def test_analyst_returns_same_probabilities(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """Analyst should not modify win/draw/lose probabilities."""
        result = system.apply_style(balanced_prediction, CoachStyleType.ANALYST)

        assert result.win_prob == pytest.approx(balanced_prediction.win_prob, abs=1e-9)
        assert result.draw_prob == pytest.approx(balanced_prediction.draw_prob, abs=1e-9)
        assert result.lose_prob == pytest.approx(balanced_prediction.lose_prob, abs=1e-9)

    def test_analyst_preserves_team_names(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """Analyst should preserve team names."""
        result = system.apply_style(balanced_prediction, CoachStyleType.ANALYST)

        assert result.team_a == balanced_prediction.team_a
        assert result.team_b == balanced_prediction.team_b

    def test_analyst_sets_style_name(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """Analyst should set coach_style to the analyst value."""
        result = system.apply_style(balanced_prediction, CoachStyleType.ANALYST)
        assert result.coach_style == "分析師"

    def test_analyst_does_not_modify_original(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """Analyst should return a copy, not mutate original."""
        original_win = balanced_prediction.win_prob
        result = system.apply_style(balanced_prediction, CoachStyleType.ANALYST)
        result.win_prob = 0.99
        assert balanced_prediction.win_prob == original_win


# ============================================================================
# TEST: CONTRARIAN STYLE (Underdog Boost)
# ============================================================================


class TestContrarianStyle:
    """Test that contrarian style boosts underdog to [35%, 40%]."""

    def test_contrarian_boosts_underdog_team_b(
        self, system: CoachStyleSystem, strong_favorite_prediction: SimplePrediction
    ):
        """When Team B is underdog with < 35%, contrarian should boost to [35%, 40%]."""
        result = system.apply_style(
            strong_favorite_prediction, CoachStyleType.CONTRARIAN
        )
        # Team B's win prob (lose_prob from Team A perspective) should be boosted
        assert 0.35 <= result.lose_prob <= 0.40

    def test_contrarian_boosts_underdog_team_a(
        self, system: CoachStyleSystem, weak_team_a_prediction: SimplePrediction
    ):
        """When Team A is underdog with < 35%, contrarian should boost to [35%, 40%]."""
        result = system.apply_style(
            weak_team_a_prediction, CoachStyleType.CONTRARIAN
        )
        # Team A's win prob should be boosted
        assert 0.35 <= result.win_prob <= 0.40

    def test_contrarian_no_boost_when_above_threshold(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """When no team is below 35%, contrarian should not boost."""
        result = system.apply_style(balanced_prediction, CoachStyleType.CONTRARIAN)
        # Both are above 35%, so no dramatic boost
        assert result.win_prob == pytest.approx(balanced_prediction.win_prob, abs=1e-9)
        assert result.lose_prob == pytest.approx(balanced_prediction.lose_prob, abs=1e-9)

    def test_contrarian_maintains_sum(
        self, system: CoachStyleSystem, strong_favorite_prediction: SimplePrediction
    ):
        """Contrarian adjustment must maintain W+D+L = 1.0."""
        result = system.apply_style(
            strong_favorite_prediction, CoachStyleType.CONTRARIAN
        )
        total = result.win_prob + result.draw_prob + result.lose_prob
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_contrarian_sets_style_name(
        self, system: CoachStyleSystem, strong_favorite_prediction: SimplePrediction
    ):
        """Contrarian should set coach_style to the contrarian value."""
        result = system.apply_style(
            strong_favorite_prediction, CoachStyleType.CONTRARIAN
        )
        assert result.coach_style == "反向思考者"

    def test_contrarian_edge_case_underdog_at_exactly_35(
        self, system: CoachStyleSystem
    ):
        """When underdog is exactly at 35%, contrarian should NOT boost."""
        pred = SimplePrediction(
            team_a="Brazil",
            team_b="Jamaica",
            win_prob=0.45,
            draw_prob=0.20,
            lose_prob=0.35,
        )
        result = system.apply_style(pred, CoachStyleType.CONTRARIAN)
        # 35% is not < 35%, so no boost
        assert result.lose_prob == pytest.approx(0.35, abs=1e-6)


# ============================================================================
# TEST: TACTICIAN STYLE (Dynamic Factors)
# ============================================================================


class TestTacticianStyle:
    """Test that tactician style adjusts based on streaks, revenge, fatigue."""

    def test_tactician_win_streak_bonus(self, system: CoachStyleSystem):
        """Win streak >= 3 should boost team's win probability."""
        pred = SimplePrediction(
            team_a="Argentina",
            team_b="France",
            win_prob=0.40,
            draw_prob=0.30,
            lose_prob=0.30,
            team_a_win_streak=4,
        )
        result = system.apply_style(pred, CoachStyleType.TACTICIAN)
        # Team A has win streak, should have higher win_prob
        assert result.win_prob > pred.win_prob

    def test_tactician_fatigue_penalty(self, system: CoachStyleSystem):
        """Fatigue (rest < 3 days) should reduce team's probability."""
        pred = SimplePrediction(
            team_a="Argentina",
            team_b="France",
            win_prob=0.40,
            draw_prob=0.30,
            lose_prob=0.30,
            team_a_days_rest=2,
        )
        result = system.apply_style(pred, CoachStyleType.TACTICIAN)
        # Team A fatigued, win_prob should decrease
        assert result.win_prob < pred.win_prob

    def test_tactician_revenge_bonus(self, system: CoachStyleSystem):
        """Revenge factor should boost team's probability."""
        pred = SimplePrediction(
            team_a="Argentina",
            team_b="France",
            win_prob=0.40,
            draw_prob=0.30,
            lose_prob=0.30,
            team_a_revenge=True,
        )
        result = system.apply_style(pred, CoachStyleType.TACTICIAN)
        # Team A seeking revenge, win_prob should increase
        assert result.win_prob > pred.win_prob

    def test_tactician_no_factors_no_change(self, system: CoachStyleSystem):
        """When no dynamic factors apply, probabilities stay approximately the same."""
        pred = SimplePrediction(
            team_a="Brazil",
            team_b="Jamaica",
            win_prob=0.50,
            draw_prob=0.25,
            lose_prob=0.25,
            team_a_win_streak=1,
            team_b_win_streak=0,
            team_a_days_rest=7,
            team_b_days_rest=7,
        )
        result = system.apply_style(pred, CoachStyleType.TACTICIAN)
        # No factors apply: no streaks >= 3, no fatigue, no revenge
        assert result.win_prob == pytest.approx(pred.win_prob, abs=1e-6)
        assert result.draw_prob == pytest.approx(pred.draw_prob, abs=1e-6)
        assert result.lose_prob == pytest.approx(pred.lose_prob, abs=1e-6)

    def test_tactician_maintains_sum(
        self, system: CoachStyleSystem, tactician_prediction: SimplePrediction
    ):
        """Tactician adjustment must maintain W+D+L = 1.0."""
        result = system.apply_style(tactician_prediction, CoachStyleType.TACTICIAN)
        total = result.win_prob + result.draw_prob + result.lose_prob
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_tactician_sets_style_name(
        self, system: CoachStyleSystem, tactician_prediction: SimplePrediction
    ):
        """Tactician should set coach_style to the tactician value."""
        result = system.apply_style(tactician_prediction, CoachStyleType.TACTICIAN)
        assert result.coach_style == "戰術家"


# ============================================================================
# TEST: PROBABILITY NORMALIZATION (All Styles)
# ============================================================================


class TestProbabilityNormalization:
    """Test that all styles maintain W+D+L = 100%."""

    @pytest.mark.parametrize("style", list(CoachStyleType))
    def test_all_styles_sum_to_one(
        self, system: CoachStyleSystem, style: CoachStyleType
    ):
        """All coach styles must produce probabilities summing to 1.0."""
        pred = SimplePrediction(
            team_a="Brazil",
            team_b="Jamaica",
            win_prob=0.65,
            draw_prob=0.20,
            lose_prob=0.15,
            team_a_win_streak=4,
            team_b_days_rest=2,
            team_a_revenge=True,
        )
        result = system.apply_style(pred, style)
        total = result.win_prob + result.draw_prob + result.lose_prob
        assert total == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("style", list(CoachStyleType))
    def test_all_probabilities_non_negative(
        self, system: CoachStyleSystem, style: CoachStyleType
    ):
        """All probabilities must be non-negative after style application."""
        pred = SimplePrediction(
            team_a="Brazil",
            team_b="Jamaica",
            win_prob=0.80,
            draw_prob=0.10,
            lose_prob=0.10,
            team_a_win_streak=5,
            team_b_loss_streak=4,
            team_b_days_rest=1,
        )
        result = system.apply_style(pred, style)
        assert result.win_prob >= 0.0
        assert result.draw_prob >= 0.0
        assert result.lose_prob >= 0.0


# ============================================================================
# TEST: NARRATIVE GENERATION
# ============================================================================


class TestNarrativeGeneration:
    """Test that narratives start with correct prefix for each style."""

    @pytest.mark.parametrize(
        "style,expected_prefix",
        [
            (CoachStyleType.ANALYST, "根據統計分析…"),
            (CoachStyleType.CONTRARIAN, "從冷門角度…"),
            (CoachStyleType.TACTICIAN, "考量戰術因素…"),
        ],
    )
    def test_narrative_starts_with_prefix(
        self,
        system: CoachStyleSystem,
        balanced_prediction: SimplePrediction,
        style: CoachStyleType,
        expected_prefix: str,
    ):
        """Narrative must start with the designated prefix for each style."""
        narrative = system.generate_narrative(style, balanced_prediction)
        assert narrative.startswith(expected_prefix)

    def test_narrative_contains_team_names(
        self, system: CoachStyleSystem, balanced_prediction: SimplePrediction
    ):
        """Narrative should reference the team names."""
        narrative = system.generate_narrative(
            CoachStyleType.ANALYST, balanced_prediction
        )
        assert "Brazil" in narrative
        assert "Argentina" in narrative

    def test_contrarian_narrative_mentions_underdog(
        self, system: CoachStyleSystem, strong_favorite_prediction: SimplePrediction
    ):
        """Contrarian narrative should mention the underdog team."""
        narrative = system.generate_narrative(
            CoachStyleType.CONTRARIAN, strong_favorite_prediction
        )
        assert "Jamaica" in narrative  # Jamaica is the underdog

    def test_tactician_narrative_mentions_factors(
        self, system: CoachStyleSystem, tactician_prediction: SimplePrediction
    ):
        """Tactician narrative should mention applicable factors."""
        narrative = system.generate_narrative(
            CoachStyleType.TACTICIAN, tactician_prediction
        )
        # Team A has win streak and revenge
        assert "連勝" in narrative or "復仇" in narrative


# ============================================================================
# TEST: KEYWORD MAPPINGS
# ============================================================================


class TestKeywordMappings:
    """Test that STYLE_KEYWORDS maps correctly."""

    def test_conservative_maps_to_analyst(self):
        assert STYLE_KEYWORDS["conservative"] == CoachStyleType.ANALYST
        assert STYLE_KEYWORDS["保守"] == CoachStyleType.ANALYST

    def test_aggressive_maps_to_contrarian(self):
        assert STYLE_KEYWORDS["aggressive"] == CoachStyleType.CONTRARIAN
        assert STYLE_KEYWORDS["激進"] == CoachStyleType.CONTRARIAN

    def test_balanced_maps_to_tactician(self):
        assert STYLE_KEYWORDS["balanced"] == CoachStyleType.TACTICIAN
        assert STYLE_KEYWORDS["平衡"] == CoachStyleType.TACTICIAN

    def test_all_styles_have_narrative_prefix(self):
        """Every CoachStyleType must have a narrative prefix."""
        for style in CoachStyleType:
            assert style in STYLE_NARRATIVE_PREFIX
