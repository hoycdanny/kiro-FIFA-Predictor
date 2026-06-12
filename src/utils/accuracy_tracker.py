"""
Accuracy Tracker for the FIFA Predictor Power.

Tracks prediction accuracy by comparing predicted results against actual
match outcomes. Provides breakdowns by coach style, confidence calibration
bands, and cross-confederation analysis.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

from dataclasses import dataclass, field
from typing import Optional

from src.data.data_manager import DataManager, MatchResult, PredictionLogEntry
from src.utils.constants import CONFEDERATION_MAP, MIN_ACCURACY_SAMPLE_SIZE


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class StyleAccuracy:
    """Per-coach-style accuracy metrics."""

    total: int
    exact_score_rate: float
    direction_rate: float
    avg_goal_error: float


@dataclass
class AccuracyReport:
    """Full accuracy report."""

    total_matches: int
    exact_score_rate: float        # Percentage of exact score hits
    direction_rate: float          # Percentage of correct W/D/L prediction
    avg_goal_error: float          # Average absolute goal difference error
    by_coach_style: dict[str, StyleAccuracy] = field(default_factory=dict)
    confidence_calibration: dict[str, float] = field(default_factory=dict)
    cross_confederation: dict[str, float] = field(default_factory=dict)


# ============================================================================
# ACCURACY TRACKER
# ============================================================================


class AccuracyTracker:
    """
    Tracks and calculates prediction accuracy metrics.

    Compares prediction log entries against actual match results to compute
    exact score hit rate, direction hit rate, average goal error, and
    breakdowns by coach style, confidence band, and confederation pair.
    """

    MIN_SAMPLE_SIZE: int = MIN_ACCURACY_SAMPLE_SIZE  # 3

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    def calculate_report(self) -> "AccuracyReport | str":
        """
        Calculate accuracy report.

        Returns:
            AccuracyReport if sufficient data (>= 3 matches with results).
            str message if zero matches or insufficient sample.
        """
        match_results = self.data_manager.load_match_results()
        predictions = self._load_predictions()

        # Match predictions to results by match_id
        paired = self._pair_predictions_with_results(predictions, match_results)

        total = len(paired)

        # No data case
        if total == 0:
            return "目前無已完賽數據，無法提供準確度數據"

        # Insufficient data case
        if total < self.MIN_SAMPLE_SIZE:
            return f"目前樣本數不足（已完成 {total} 場），無法提供具統計意義的準確度數據"

        # Compute overall metrics
        exact_score_rate = self._calc_exact_score_rate(paired)
        direction_rate = self._calc_direction_rate(paired)
        avg_goal_error = self._calc_avg_goal_error(paired)

        # Per coach style breakdown
        by_coach_style = self._calc_by_coach_style(paired)

        # Confidence calibration (3 bands)
        confidence_calibration = self._calc_confidence_calibration(paired)

        # Cross-confederation analysis
        cross_confederation = self._calc_cross_confederation(paired)

        return AccuracyReport(
            total_matches=total,
            exact_score_rate=exact_score_rate,
            direction_rate=direction_rate,
            avg_goal_error=avg_goal_error,
            by_coach_style=by_coach_style,
            confidence_calibration=confidence_calibration,
            cross_confederation=cross_confederation,
        )

    def record_prediction_vs_actual(
        self, prediction: PredictionLogEntry, actual: MatchResult
    ) -> None:
        """
        Record comparison between prediction and actual result.

        This is a convenience method; the actual comparison is done
        in calculate_report() by matching predictions_log against match_results.
        The DataManager handles persistence.
        """
        # The prediction log entry is already stored via DataManager.append_prediction_log()
        # The match result is already stored via DataManager.save_match_result()
        # This method exists for explicit recording if needed in future extensions.
        pass

    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================

    def _load_predictions(self) -> list[PredictionLogEntry]:
        """Load all prediction log entries from data."""
        import json
        from pathlib import Path

        filepath = self.data_manager.data_dir / "predictions_log.json"
        try:
            data = self.data_manager._read_json(filepath)
        except Exception:
            return []

        entries: list[PredictionLogEntry] = []
        for item in data.get("predictions", []):
            entry = PredictionLogEntry(
                timestamp=item["timestamp"],
                match_id=item["match_id"],
                team_a=item["team_a"],
                team_b=item["team_b"],
                predicted_score=tuple(item["predicted_score"]),
                win_draw_lose=tuple(item["win_draw_lose"]),
                confidence_index=item["confidence_index"],
                coach_style=item["coach_style"],
                model_weights=item.get("model_weights", {}),
            )
            entries.append(entry)

        return entries

    def _pair_predictions_with_results(
        self,
        predictions: list[PredictionLogEntry],
        results: list[MatchResult],
    ) -> list[tuple[PredictionLogEntry, MatchResult]]:
        """
        Pair predictions with their corresponding match results by match_id.

        Only returns pairs where both prediction and result exist.
        If multiple predictions exist for the same match_id, uses the latest one.
        """
        # Build result lookup by match_id
        result_map: dict[str, MatchResult] = {}
        for r in results:
            result_map[r.match_id] = r

        # Build prediction lookup by match_id (latest prediction wins)
        pred_map: dict[str, PredictionLogEntry] = {}
        for p in predictions:
            if p.match_id not in pred_map:
                pred_map[p.match_id] = p
            else:
                # Keep the latest one by timestamp
                if p.timestamp > pred_map[p.match_id].timestamp:
                    pred_map[p.match_id] = p

        # Pair them
        paired: list[tuple[PredictionLogEntry, MatchResult]] = []
        for match_id, pred in pred_map.items():
            if match_id in result_map:
                paired.append((pred, result_map[match_id]))

        return paired

    def _get_predicted_direction(self, prediction: PredictionLogEntry) -> str:
        """
        Determine predicted direction from win/draw/lose probabilities.

        Returns: "win_a", "draw", or "win_b"
        """
        win_prob, draw_prob, lose_prob = prediction.win_draw_lose
        if win_prob >= draw_prob and win_prob >= lose_prob:
            return "win_a"
        elif draw_prob >= win_prob and draw_prob >= lose_prob:
            return "draw"
        else:
            return "win_b"

    def _get_actual_direction(self, result: MatchResult) -> str:
        """
        Determine actual direction from match score.

        Returns: "win_a", "draw", or "win_b"
        """
        if result.score_a > result.score_b:
            return "win_a"
        elif result.score_a == result.score_b:
            return "draw"
        else:
            return "win_b"

    def _calc_exact_score_rate(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> float:
        """
        Calculate exact score hit rate as percentage.

        exact_score_rate = count(predicted_score == actual_score) / total × 100
        """
        total = len(paired)
        if total == 0:
            return 0.0

        hits = sum(
            1
            for pred, result in paired
            if pred.predicted_score[0] == result.score_a
            and pred.predicted_score[1] == result.score_b
        )

        return round(hits / total * 100, 1)

    def _calc_direction_rate(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> float:
        """
        Calculate direction hit rate as percentage.

        direction_rate = count(predicted_direction == actual_direction) / total × 100
        """
        total = len(paired)
        if total == 0:
            return 0.0

        hits = sum(
            1
            for pred, result in paired
            if self._get_predicted_direction(pred) == self._get_actual_direction(result)
        )

        return round(hits / total * 100, 1)

    def _calc_avg_goal_error(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> float:
        """
        Calculate average absolute goal error.

        avg_goal_error = sum(|predicted_total_goals - actual_total_goals|) / total
        """
        total = len(paired)
        if total == 0:
            return 0.0

        total_error = sum(
            abs(
                (pred.predicted_score[0] + pred.predicted_score[1])
                - (result.score_a + result.score_b)
            )
            for pred, result in paired
        )

        return round(total_error / total, 2)

    def _calc_by_coach_style(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> dict[str, StyleAccuracy]:
        """
        Calculate accuracy metrics grouped by coach style.

        Groups predictions by their coach_style field and computes
        exact_score_rate, direction_rate, and avg_goal_error for each.
        """
        # Group by coach style
        style_groups: dict[str, list[tuple[PredictionLogEntry, MatchResult]]] = {}
        for pred, result in paired:
            style = pred.coach_style
            if style not in style_groups:
                style_groups[style] = []
            style_groups[style].append((pred, result))

        # Calculate metrics per style
        by_style: dict[str, StyleAccuracy] = {}
        for style, group in style_groups.items():
            by_style[style] = StyleAccuracy(
                total=len(group),
                exact_score_rate=self._calc_exact_score_rate(group),
                direction_rate=self._calc_direction_rate(group),
                avg_goal_error=self._calc_avg_goal_error(group),
            )

        return by_style

    def _calc_confidence_calibration(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> dict[str, float]:
        """
        Calculate direction hit rate per confidence band.

        Bands:
        - "0-33" (low confidence)
        - "34-66" (medium confidence)
        - "67-100" (high confidence)
        """
        bands: dict[str, list[tuple[PredictionLogEntry, MatchResult]]] = {
            "0-33": [],
            "34-66": [],
            "67-100": [],
        }

        for pred, result in paired:
            ci = pred.confidence_index
            if ci <= 33:
                bands["0-33"].append((pred, result))
            elif ci <= 66:
                bands["34-66"].append((pred, result))
            else:
                bands["67-100"].append((pred, result))

        calibration: dict[str, float] = {}
        for band_name, group in bands.items():
            if len(group) > 0:
                hits = sum(
                    1
                    for pred, result in group
                    if self._get_predicted_direction(pred)
                    == self._get_actual_direction(result)
                )
                calibration[band_name] = round(hits / len(group) * 100, 1)
            else:
                calibration[band_name] = 0.0

        return calibration

    def _calc_cross_confederation(
        self, paired: list[tuple[PredictionLogEntry, MatchResult]]
    ) -> dict[str, float]:
        """
        Calculate direction hit rate per confederation pair.

        Groups matches by the confederation pair (e.g. "UEFA vs CONMEBOL")
        and computes direction rate for each pair.
        """
        conf_groups: dict[str, list[tuple[PredictionLogEntry, MatchResult]]] = {}

        for pred, result in paired:
            conf_a = CONFEDERATION_MAP.get(result.team_a, "Unknown")
            conf_b = CONFEDERATION_MAP.get(result.team_b, "Unknown")

            # Create a consistent pair key (alphabetical order)
            pair = tuple(sorted([conf_a, conf_b]))
            pair_key = f"{pair[0]} vs {pair[1]}"

            if pair_key not in conf_groups:
                conf_groups[pair_key] = []
            conf_groups[pair_key].append((pred, result))

        # Calculate direction rate per confederation pair
        cross_conf: dict[str, float] = {}
        for pair_key, group in conf_groups.items():
            if len(group) > 0:
                hits = sum(
                    1
                    for pred, result in group
                    if self._get_predicted_direction(pred)
                    == self._get_actual_direction(result)
                )
                cross_conf[pair_key] = round(hits / len(group) * 100, 1)

        return cross_conf
