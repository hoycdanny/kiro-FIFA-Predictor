"""
Accuracy stats MCP tool handler.

Thin wrapper that queries the AccuracyTracker and formats the report.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.output.formatter import OutputFormatter
    from src.utils.accuracy_tracker import AccuracyTracker


async def handle_accuracy_stats(
    accuracy_tracker: Optional["AccuracyTracker"],
    formatter: Optional["OutputFormatter"],
) -> str:
    """Handle the accuracy_stats MCP tool invocation.

    Args:
        accuracy_tracker: Accuracy tracker instance.
        formatter: Output formatter instance.

    Returns:
        Formatted accuracy report or insufficient data message.
    """
    if accuracy_tracker is None or formatter is None:
        return "Error: Server not initialized. Please restart the server."

    report = accuracy_tracker.calculate_report()

    # If the report is a string message (insufficient data), return it directly
    if isinstance(report, str):
        return report

    # Format the full accuracy report
    return formatter.format_accuracy_report(report)
