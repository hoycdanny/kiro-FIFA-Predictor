"""
Predict group MCP tool handler.

Thin wrapper that validates group ID, executes group prediction,
and formats the standings table output.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.engine.prediction_engine import PredictionEngine
    from src.output.formatter import OutputFormatter
    from src.utils.validator import InputValidator


async def handle_predict_group(
    group_id: str,
    engine: Optional["PredictionEngine"],
    validator: Optional["InputValidator"],
    formatter: Optional["OutputFormatter"],
) -> str:
    """Handle the predict_group MCP tool invocation.

    Args:
        group_id: User-provided group identifier (A-L).
        engine: Prediction engine instance.
        validator: Input validator instance.
        formatter: Output formatter instance.

    Returns:
        Formatted group standings table or error message.
    """
    if engine is None or validator is None or formatter is None:
        return "Error: Server not initialized. Please restart the server."

    # Validate group ID
    result = validator.validate_group(group_id)
    if not isinstance(result, str):
        return _format_error(result)

    # Execute group prediction
    try:
        group_prediction = engine.predict_group(result)
    except ValueError as e:
        return f"Error: {e}"

    # Format output
    return formatter.format_group_standings(group_prediction)


def _format_error(error) -> str:
    """Format a PredictionError into a user-friendly string."""
    lines = [f"❌ {error.message}"]
    if error.suggestions:
        lines.append("有效小組代號：")
        for s in error.suggestions:
            lines.append(f"  • {s}")
    return "\n".join(lines)
