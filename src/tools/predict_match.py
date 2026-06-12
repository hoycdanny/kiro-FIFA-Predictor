"""
Predict match MCP tool handler.

Thin wrapper that validates input, executes prediction via the engine,
and formats the output. Displays all three coach style perspectives
as required by Requirement 8.4.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 8.4, 8.5
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data.data_manager import DataManager
    from src.engine.prediction_engine import PredictionEngine
    from src.output.formatter import OutputFormatter
    from src.utils.validator import InputValidator

from src.data.data_manager import PredictionError, PredictionLogEntry
from src.engine.coach_style import CoachStyleType, STYLE_NARRATIVE_PREFIX


async def handle_predict_match(
    team_a: str,
    team_b: str,
    coach_style: Optional[str],
    engine: Optional["PredictionEngine"],
    validator: Optional["InputValidator"],
    formatter: Optional["OutputFormatter"],
    data_manager: Optional["DataManager"],
) -> str:
    """Handle the predict_match MCP tool invocation.

    Steps:
    1. Validate team_a and team_b using validator.validate_team()
    2. If validation fails, return error message with suggestions
    3. If coach_style provided, validate using validator.validate_coach_style()
    4. Call engine.predict_match() for the primary style (default: analyst)
    5. Call engine.predict_match() for ALL three styles to display together
    6. Format output using formatter.format_match_prediction() for primary
    7. Append coach style comparison section showing all three perspectives
    8. Log the prediction using data_manager.append_prediction_log()
    9. Return formatted string

    Args:
        team_a: User-provided team A name.
        team_b: User-provided team B name.
        coach_style: Optional coach style string.
        engine: Prediction engine instance.
        validator: Input validator instance.
        formatter: Output formatter instance.
        data_manager: Data manager instance.

    Returns:
        Formatted prediction result string or error message.
    """
    if engine is None or validator is None or formatter is None:
        return "Error: Server not initialized. Please restart the server."

    # Step 1: Validate team A
    result_a = validator.validate_team(team_a)
    if isinstance(result_a, PredictionError):
        return _format_error(result_a)

    # Step 2: Validate team B
    result_b = validator.validate_team(team_b)
    if isinstance(result_b, PredictionError):
        return _format_error(result_b)

    # Check if any team is a non-participant (friendly match)
    is_friendly = validator.is_non_participant(result_a) or validator.is_non_participant(result_b)

    # Step 3: Validate coach style if provided
    resolved_style: Optional[str] = None
    if coach_style is not None:
        style_result = validator.validate_coach_style(coach_style)
        if isinstance(style_result, PredictionError):
            return _format_error(style_result)
        # style_result is a CoachStyleType enum; use its Chinese value
        resolved_style = style_result.value

    # Default coach style is analyst (Requirement 8.5)
    primary_style = resolved_style if resolved_style else CoachStyleType.ANALYST.value

    # Step 4: Execute prediction with primary style
    try:
        primary_prediction = engine.predict_match(
            team_a=result_a,
            team_b=result_b,
            coach_style=primary_style,
        )
    except ValueError as e:
        return f"Error: {e}"

    # Step 5: Format the primary prediction output
    formatted_output = formatter.format_match_prediction(primary_prediction)

    # Add friendly match disclaimer if applicable
    if is_friendly:
        non_participant_names = []
        if validator.is_non_participant(result_a):
            non_participant_names.append(result_a)
        if validator.is_non_participant(result_b):
            non_participant_names.append(result_b)
        disclaimer = (
            "\n\n> ⚠️ **友誼賽/熱身賽預測提示：** "
            f"{'、'.join(non_participant_names)} 非 2026 世界盃參賽隊伍，"
            "本預測使用該隊伍的估算數據（基於 FIFA 排名與近期表現），"
            "準確度可能較正式參賽隊伍的預測為低。"
        )
        formatted_output = formatted_output + disclaimer

    # Step 6: Generate all three coach style perspectives (Requirement 8.4)
    coach_styles_section = _generate_all_coach_styles(
        engine=engine,
        team_a=result_a,
        team_b=result_b,
        primary_style=primary_style,
    )

    # Combine: primary prediction + all three styles section
    full_output = f"{formatted_output}\n\n{coach_styles_section}"

    # Step 7: Log the prediction (if data_manager available)
    if data_manager is not None:
        try:
            _log_prediction(
                data_manager=data_manager,
                prediction=primary_prediction,
                team_a=result_a,
                team_b=result_b,
            )
        except Exception:
            # Logging failure should not break the response
            pass

    return full_output


def _generate_all_coach_styles(
    engine: "PredictionEngine",
    team_a: str,
    team_b: str,
    primary_style: str,
) -> str:
    """Generate the three coach style comparison section.

    Requirement 8.4: Display all three Coach_Style predictions,
    each with adjusted win rate, recommended score, and narrative text.

    Args:
        engine: Prediction engine instance.
        team_a: Canonical team A name.
        team_b: Canonical team B name.
        primary_style: The primary coach style used (for highlighting).

    Returns:
        Markdown-formatted string with all three perspectives.
    """
    lines: list[str] = []
    lines.append("### 🎭 Coach Style Perspectives")
    lines.append("")

    styles = [
        (CoachStyleType.ANALYST, "分析師 (Analyst)"),
        (CoachStyleType.CONTRARIAN, "反向思考者 (Contrarian)"),
        (CoachStyleType.TACTICIAN, "戰術家 (Tactician)"),
    ]

    for style_type, style_label in styles:
        try:
            prediction = engine.predict_match(
                team_a=team_a,
                team_b=team_b,
                coach_style=style_type.value,
            )

            # Mark the active/primary style
            marker = " ⬅️" if style_type.value == primary_style else ""

            lines.append(f"**{style_label}{marker}**")
            lines.append("")

            # Narrative prefix (Requirement 8.7)
            narrative = engine.coach_style.generate_narrative(
                style_type,
                _prediction_to_simple(prediction, engine, team_a, team_b),
            )
            lines.append(f"> {narrative}")
            lines.append("")

            # Adjusted win rates
            lines.append(
                f"- Win: {prediction.win_prob * 100:.1f}% | "
                f"Draw: {prediction.draw_prob * 100:.1f}% | "
                f"Lose: {prediction.lose_prob * 100:.1f}%"
            )

            # Recommended score (top 1)
            if prediction.top_scores:
                score_a, score_b, prob = prediction.top_scores[0]
                lines.append(
                    f"- Recommended Score: {score_a}-{score_b} ({prob * 100:.1f}%)"
                )

            lines.append("")

        except (ValueError, Exception):
            lines.append(f"**{style_label}**")
            lines.append("")
            lines.append("> Unable to generate this perspective.")
            lines.append("")

    return "\n".join(lines)


def _prediction_to_simple(
    prediction: "MatchPrediction",
    engine: "PredictionEngine",
    team_a: str,
    team_b: str,
) -> "SimplePrediction":
    """Convert a MatchPrediction to SimplePrediction for narrative generation.

    Args:
        prediction: The match prediction result.
        engine: Prediction engine for team data access.
        team_a: Team A name.
        team_b: Team B name.

    Returns:
        SimplePrediction suitable for CoachStyleSystem.generate_narrative().
    """
    from src.engine.coach_style import SimplePrediction
    from src.engine.prediction_engine import MatchPrediction

    # Get team profiles for dynamic data
    try:
        profile_a = engine._get_team(team_a)
        profile_b = engine._get_team(team_b)
        team_a_win_streak = profile_a.current_win_streak
        team_a_loss_streak = profile_a.current_loss_streak
        team_b_win_streak = profile_b.current_win_streak
        team_b_loss_streak = profile_b.current_loss_streak
        team_a_days_rest = engine._get_days_rest(profile_a)
        team_b_days_rest = engine._get_days_rest(profile_b)
        team_a_revenge = (profile_a.eliminated_by_2022 == team_b)
        team_b_revenge = (profile_b.eliminated_by_2022 == team_a)
    except (ValueError, AttributeError):
        team_a_win_streak = 0
        team_a_loss_streak = 0
        team_b_win_streak = 0
        team_b_loss_streak = 0
        team_a_days_rest = 7
        team_b_days_rest = 7
        team_a_revenge = False
        team_b_revenge = False

    return SimplePrediction(
        team_a=team_a,
        team_b=team_b,
        win_prob=prediction.win_prob,
        draw_prob=prediction.draw_prob,
        lose_prob=prediction.lose_prob,
        top_scores=prediction.top_scores,
        confidence_index=prediction.confidence_index,
        over_2_5=prediction.over_2_5,
        under_2_5=prediction.under_2_5,
        expected_goals_a=prediction.expected_goals_a,
        expected_goals_b=prediction.expected_goals_b,
        coach_style=prediction.coach_style,
        team_a_win_streak=team_a_win_streak,
        team_a_loss_streak=team_a_loss_streak,
        team_b_win_streak=team_b_win_streak,
        team_b_loss_streak=team_b_loss_streak,
        team_a_days_rest=team_a_days_rest,
        team_b_days_rest=team_b_days_rest,
        team_a_revenge=team_a_revenge,
        team_b_revenge=team_b_revenge,
    )


def _log_prediction(
    data_manager: "DataManager",
    prediction: "MatchPrediction",
    team_a: str,
    team_b: str,
) -> None:
    """Log the prediction to predictions_log.json.

    Requirement 9.6: Log prediction with timestamp, match ID,
    results, and model parameters.

    Args:
        data_manager: Data manager instance.
        prediction: The match prediction result.
        team_a: Team A canonical name.
        team_b: Team B canonical name.
    """
    from src.engine.prediction_engine import MatchPrediction

    # Generate a match ID from team names
    match_id = f"{team_a}_vs_{team_b}"

    # Use top score as predicted score
    predicted_score = (0, 0)
    if prediction.top_scores:
        predicted_score = (prediction.top_scores[0][0], prediction.top_scores[0][1])

    log_entry = PredictionLogEntry(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        match_id=match_id,
        team_a=team_a,
        team_b=team_b,
        predicted_score=predicted_score,
        win_draw_lose=(
            prediction.win_prob,
            prediction.draw_prob,
            prediction.lose_prob,
        ),
        confidence_index=prediction.confidence_index,
        coach_style=prediction.coach_style,
        model_weights={},  # Weights are internal to the ensemble
    )

    data_manager.append_prediction_log(log_entry)


def _format_error(error: PredictionError) -> str:
    """Format a PredictionError into a user-friendly string."""
    lines = [f"❌ {error.message}"]
    if error.suggestions:
        lines.append("建議：")
        for s in error.suggestions:
            lines.append(f"  • {s}")
    return "\n".join(lines)
