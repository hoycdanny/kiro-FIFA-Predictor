"""
Team info MCP tool handler.

Thin wrapper that resolves team name via fuzzy matching and formats
the team profile output.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data.data_manager import DataManager, PredictionError
    from src.output.formatter import OutputFormatter
    from src.utils.validator import InputValidator


async def handle_team_info(
    team_name: str,
    validator: Optional["InputValidator"],
    formatter: Optional["OutputFormatter"],
    data_manager: Optional["DataManager"],
) -> str:
    """Handle the team_info MCP tool invocation.

    Resolves team name via TeamMatcher (fuzzy matching), handles ambiguous
    matches by listing all candidates for user selection, and formats the
    team profile output when a single match is found.

    Args:
        team_name: User-provided team name (supports fuzzy matching).
        validator: Input validator instance.
        formatter: Output formatter instance.
        data_manager: Data manager instance.

    Returns:
        Formatted team profile or error with suggestions.
    """
    if validator is None or formatter is None or data_manager is None:
        return "Error: Server not initialized. Please restart the server."

    # Validate and resolve team name
    result = validator.validate_team(team_name)

    if not isinstance(result, str):
        # It's a PredictionError — distinguish ambiguous vs no-match
        return _format_error(result, team_name)

    # Canonical name resolved — load team profile
    teams = data_manager.load_teams()
    team_profile = None
    for team in teams:
        if team.name == result:
            team_profile = team
            break

    # If not found in participants, check non-participant teams
    if team_profile is None:
        np_teams = data_manager.load_non_participant_teams()
        for team in np_teams:
            if team.name == result:
                team_profile = team
                break

    if team_profile is None:
        return f"❌ 球隊「{result}」的資料不在資料庫中。"

    # Format output using the configured renderer
    output = formatter.format_team_profile(team_profile)

    # Add non-participant disclaimer
    if team_profile.group == "N/A":
        output += (
            "\n\n> ⚠️ 此球隊非 2026 世界盃參賽隊伍，"
            "資料為基於 FIFA 排名與近期表現的估算值。"
        )

    return output


def _format_error(error: "PredictionError", original_query: str) -> str:
    """Format a PredictionError into a user-friendly message.

    Distinguishes between:
    - Ambiguous match (multiple candidates): lists all for user selection (Req 6.4)
    - No match: shows error with up to 3 suggestions (Req 6.3)

    Args:
        error: The PredictionError from validation.
        original_query: The original team name query for context.

    Returns:
        Formatted error string.
    """
    lines: list[str] = [f"❌ {error.message}"]

    if error.error_code == "INVALID_TEAM" and error.suggestions:
        # Check if this is an ambiguous match (multiple candidates) or no match
        # The validator sets a specific message pattern for multiple matches
        if "多個相似結果" in error.message:
            # Requirement 6.4: List all matches for user selection
            lines.append("")
            lines.append("請從以下球隊中選擇：")
            for i, candidate in enumerate(error.suggestions, 1):
                lines.append(f"  {i}. {candidate}")
        else:
            # Requirement 6.3: No match — show up to 3 suggestions
            lines.append("")
            lines.append("您是否要找以下球隊？")
            for suggestion in error.suggestions[:3]:
                lines.append(f"  • {suggestion}")
    elif error.suggestions:
        lines.append("")
        lines.append("建議：")
        for s in error.suggestions:
            lines.append(f"  • {s}")

    return "\n".join(lines)
