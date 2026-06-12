"""
Property-based tests for streak counter update correctness.

Tests Property 20 from the design document:
- Property 20: Streak counter update correctness

Validates: Requirements 4.5
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from src.data.data_manager import TeamProfile


# ============================================================================
# Strategies
# ============================================================================

# Match outcome: "win", "loss", or "draw"
match_outcomes = st.sampled_from(["win", "loss", "draw"])

# Sequences of match outcomes for a team
match_outcome_sequences = st.lists(
    match_outcomes,
    min_size=1,
    max_size=50,
)


def _make_team_profile() -> TeamProfile:
    """Create a minimal TeamProfile for streak testing."""
    return TeamProfile(
        name="TestTeam",
        name_zh="測試隊",
        aliases=["TST"],
        confederation="UEFA",
        fifa_ranking=1,
        fifa_points=1800.0,
        elo_rating=2000,
        group="A",
        recent_goals_avg=1.5,
        recent_conceded_avg=0.8,
        recent_win_rate=60.0,
        recent_draw_rate=20.0,
        recent_loss_rate=20.0,
        neutral_win_rate=55.0,
        best_wc_result="Champion",
        vs_top20_win_rate=50.0,
        wc_first_match_win_rate=70.0,
        penalty_shootout_win_rate=50.0,
        first_half_goal_pct=45.0,
        second_half_goal_pct=55.0,
        clean_sheet_rate=30.0,
        failed_to_score_rate=10.0,
        current_win_streak=0,
        current_loss_streak=0,
        last_match_date=None,
    )


# ============================================================================
# Property 20: Streak counter update correctness
# ============================================================================


class TestStreakCounterUpdateCorrectness:
    """
    Property 20: Streak counter update correctness

    For any sequence of match results for a team, after processing each result:
    - if the team won, win_streak increments by 1 and loss_streak resets to 0
    - if the team lost, loss_streak increments by 1 and win_streak resets to 0
    - if draw, both streaks reset to 0

    **Validates: Requirements 4.5**
    """

    @given(outcomes=match_outcome_sequences)
    @settings(max_examples=300)
    def test_streak_updates_match_specification(
        self, outcomes: list[str]
    ) -> None:
        """
        For any sequence of outcomes, after each result the streaks
        SHALL match the specification exactly.

        **Validates: Requirements 4.5**
        """
        from src.tools.update_results import RecalibrationProcess

        team = _make_team_profile()

        for outcome in outcomes:
            # Determine scores based on outcome
            if outcome == "win":
                goals_for, goals_against = 2, 0
            elif outcome == "loss":
                goals_for, goals_against = 0, 2
            else:  # draw
                goals_for, goals_against = 1, 1

            # Save previous state for assertions
            prev_win_streak = team.current_win_streak
            prev_loss_streak = team.current_loss_streak

            # Call the actual streak update method
            RecalibrationProcess._update_team_streak(
                None, team, goals_for, goals_against, "2025-06-15"
            )

            # Verify according to specification
            if outcome == "win":
                assert team.current_win_streak == prev_win_streak + 1, (
                    f"After win: expected win_streak={prev_win_streak + 1}, "
                    f"got {team.current_win_streak}"
                )
                assert team.current_loss_streak == 0, (
                    f"After win: expected loss_streak=0, "
                    f"got {team.current_loss_streak}"
                )
            elif outcome == "loss":
                assert team.current_loss_streak == prev_loss_streak + 1, (
                    f"After loss: expected loss_streak={prev_loss_streak + 1}, "
                    f"got {team.current_loss_streak}"
                )
                assert team.current_win_streak == 0, (
                    f"After loss: expected win_streak=0, "
                    f"got {team.current_win_streak}"
                )
            else:  # draw
                assert team.current_win_streak == 0, (
                    f"After draw: expected win_streak=0, "
                    f"got {team.current_win_streak}"
                )
                assert team.current_loss_streak == 0, (
                    f"After draw: expected loss_streak=0, "
                    f"got {team.current_loss_streak}"
                )

    @given(outcomes=match_outcome_sequences)
    @settings(max_examples=300)
    def test_win_streak_counts_consecutive_wins(
        self, outcomes: list[str]
    ) -> None:
        """
        After processing a full sequence, win_streak SHALL equal the
        number of consecutive wins at the end of the sequence.

        **Validates: Requirements 4.5**
        """
        from src.tools.update_results import RecalibrationProcess

        team = _make_team_profile()

        # Process all outcomes
        for outcome in outcomes:
            if outcome == "win":
                goals_for, goals_against = 3, 1
            elif outcome == "loss":
                goals_for, goals_against = 0, 1
            else:
                goals_for, goals_against = 2, 2

            RecalibrationProcess._update_team_streak(
                None, team, goals_for, goals_against, "2025-06-15"
            )

        # Calculate expected final win streak
        expected_win_streak = 0
        for outcome in reversed(outcomes):
            if outcome == "win":
                expected_win_streak += 1
            else:
                break

        assert team.current_win_streak == expected_win_streak, (
            f"Expected final win_streak={expected_win_streak}, "
            f"got {team.current_win_streak} for sequence ending: "
            f"{outcomes[-5:]}"
        )

    @given(outcomes=match_outcome_sequences)
    @settings(max_examples=300)
    def test_loss_streak_counts_consecutive_losses(
        self, outcomes: list[str]
    ) -> None:
        """
        After processing a full sequence, loss_streak SHALL equal the
        number of consecutive losses at the end of the sequence.

        **Validates: Requirements 4.5**
        """
        from src.tools.update_results import RecalibrationProcess

        team = _make_team_profile()

        # Process all outcomes
        for outcome in outcomes:
            if outcome == "win":
                goals_for, goals_against = 1, 0
            elif outcome == "loss":
                goals_for, goals_against = 0, 3
            else:
                goals_for, goals_against = 0, 0

            RecalibrationProcess._update_team_streak(
                None, team, goals_for, goals_against, "2025-06-15"
            )

        # Calculate expected final loss streak
        expected_loss_streak = 0
        for outcome in reversed(outcomes):
            if outcome == "loss":
                expected_loss_streak += 1
            else:
                break

        assert team.current_loss_streak == expected_loss_streak, (
            f"Expected final loss_streak={expected_loss_streak}, "
            f"got {team.current_loss_streak} for sequence ending: "
            f"{outcomes[-5:]}"
        )

    @given(
        initial_win_streak=st.integers(min_value=0, max_value=20),
        initial_loss_streak=st.integers(min_value=0, max_value=20),
        outcome=match_outcomes,
    )
    @settings(max_examples=200)
    def test_streak_update_from_any_initial_state(
        self, initial_win_streak: int, initial_loss_streak: int, outcome: str
    ) -> None:
        """
        Regardless of initial streak state, a single outcome SHALL
        update streaks according to the specification.

        **Validates: Requirements 4.5**
        """
        from src.tools.update_results import RecalibrationProcess

        team = _make_team_profile()
        team.current_win_streak = initial_win_streak
        team.current_loss_streak = initial_loss_streak

        if outcome == "win":
            goals_for, goals_against = 4, 2
        elif outcome == "loss":
            goals_for, goals_against = 1, 3
        else:
            goals_for, goals_against = 2, 2

        RecalibrationProcess._update_team_streak(
            None, team, goals_for, goals_against, "2025-07-01"
        )

        if outcome == "win":
            assert team.current_win_streak == initial_win_streak + 1
            assert team.current_loss_streak == 0
        elif outcome == "loss":
            assert team.current_loss_streak == initial_loss_streak + 1
            assert team.current_win_streak == 0
        else:
            assert team.current_win_streak == 0
            assert team.current_loss_streak == 0
