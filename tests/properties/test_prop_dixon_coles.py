"""
Property-based tests for the Dixon-Coles Poisson Model.

**Validates: Requirements 7.1**
"""

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import composite, floats, sampled_from

from src.data.data_manager import TeamProfile
from src.engine.dixon_coles import DixonColesModel


CONFEDERATIONS = ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]


@composite
def team_profile_for_dixon_coles(draw):
    """Generate a TeamProfile with valid parameters for Dixon-Coles model."""
    recent_goals_avg = draw(floats(min_value=0.1, max_value=5.0))
    recent_conceded_avg = draw(floats(min_value=0.1, max_value=5.0))
    confederation = draw(sampled_from(CONFEDERATIONS))

    return TeamProfile(
        name="TestTeam",
        name_zh="测试队",
        aliases=[],
        confederation=confederation,
        fifa_ranking=1,
        fifa_points=1500.0,
        elo_rating=1500,
        group="A",
        recent_goals_avg=recent_goals_avg,
        recent_conceded_avg=recent_conceded_avg,
        recent_win_rate=50.0,
        recent_draw_rate=25.0,
        recent_loss_rate=25.0,
        neutral_win_rate=50.0,
        best_wc_result="Group stage",
        vs_top20_win_rate=30.0,
        wc_first_match_win_rate=50.0,
        penalty_shootout_win_rate=50.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=20.0,
        failed_to_score_rate=15.0,
    )


class TestDixonColesScoreMatrixValidity:
    """
    Property 11: Dixon-Coles score matrix validity.

    For any valid pair of team parameters (attack_strength > 0,
    defense_weakness > 0), the 5×5 score probability matrix SHALL have
    all non-negative entries and a total sum ≤ 1.0 (the remainder accounts
    for scores outside 0-4 range).

    **Validates: Requirements 7.1**
    """

    @given(
        team_a=team_profile_for_dixon_coles(),
        team_b=team_profile_for_dixon_coles(),
    )
    @settings(max_examples=200)
    def test_all_entries_non_negative(self, team_a: TeamProfile, team_b: TeamProfile):
        """All entries in the 5x5 score matrix must be non-negative."""
        model = DixonColesModel()
        matrix = model.predict(team_a, team_b)

        assert matrix.shape == (5, 5), f"Expected shape (5,5), got {matrix.shape}"
        assert np.all(matrix >= 0), (
            f"Found negative entries in score matrix: {matrix[matrix < 0]}"
        )

    @given(
        team_a=team_profile_for_dixon_coles(),
        team_b=team_profile_for_dixon_coles(),
    )
    @settings(max_examples=200)
    def test_total_sum_at_most_one(self, team_a: TeamProfile, team_b: TeamProfile):
        """The total sum of the 5x5 matrix must be ≤ 1.0."""
        model = DixonColesModel()
        matrix = model.predict(team_a, team_b)

        total = np.sum(matrix)
        assert total <= 1.0 + 1e-9, (
            f"Score matrix sum {total} exceeds 1.0"
        )

    @given(
        team_a=team_profile_for_dixon_coles(),
        team_b=team_profile_for_dixon_coles(),
    )
    @settings(max_examples=200)
    def test_matrix_shape_is_5x5(self, team_a: TeamProfile, team_b: TeamProfile):
        """The score matrix must always be 5x5."""
        model = DixonColesModel()
        matrix = model.predict(team_a, team_b)

        assert matrix.shape == (5, 5), f"Expected shape (5,5), got {matrix.shape}"
