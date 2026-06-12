"""
Recalibration Process for the FIFA Predictor Power.

Handles post-match result updates, model weight recalibration,
dynamic factor updates (streaks, fatigue, revenge), and accuracy tracking.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data.data_manager import DataManager, MatchResult, PredictionLogEntry, TeamProfile
from src.engine.ensemble import EnsembleModel, EnsembleWeights
from src.utils.constants import MAX_WEIGHT_ADJUSTMENT, WEIGHT_MAX, WEIGHT_MIN


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class RecalibrationReport:
    """Report generated after a recalibration cycle."""

    matches_evaluated: int
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    exact_score_hits: int
    direction_hits: int
    avg_goal_error: float
    systematic_bias: Optional[str] = None  # Cross-confederation bias report


@dataclass
class MatchComparison:
    """Comparison between a prediction and the actual result."""

    match_id: str
    predicted_score: tuple[int, int]
    actual_score: tuple[int, int]
    predicted_direction: str  # "win_a", "draw", "win_b"
    actual_direction: str
    exact_score_hit: bool
    direction_hit: bool
    goal_error: float  # |predicted_total - actual_total|


# ============================================================================
# RECALIBRATION PROCESS
# ============================================================================


class RecalibrationProcess:
    """Manages post-match recalibration of model weights and dynamic factors.

    After each match result is recorded, this process:
    1. Compares predictions vs actual results
    2. Evaluates sub-model performance
    3. Adjusts ensemble weights (bounded ±0.05, range [0.10, 0.60], sum=1.00)
    4. Updates team dynamic factors (streaks, fatigue, revenge)
    5. Generates a recalibration report
    """

    MAX_WEIGHT_ADJUSTMENT: float = MAX_WEIGHT_ADJUSTMENT  # 0.05
    WEIGHT_MIN: float = WEIGHT_MIN  # 0.10
    WEIGHT_MAX: float = WEIGHT_MAX  # 0.60
    TIMEOUT_SECONDS: int = 30
    BIAS_THRESHOLD_MATCHES: int = 5
    BIAS_DIRECTION_THRESHOLD: float = 0.50  # 50%

    def __init__(
        self,
        data_manager: DataManager,
        ensemble: EnsembleModel,
    ):
        self.data_manager = data_manager
        self.ensemble = ensemble

    async def update_results(
        self,
        match_results: Optional[list[MatchResult]] = None,
    ) -> RecalibrationReport:
        """Execute result update and recalibration.

        Flow:
        1. Load recent match results (or use provided ones)
        2. Load predictions log to find corresponding predictions
        3. Compare predicted vs actual for each result
        4. Calculate performance metrics per sub-model
        5. Adjust weights using _adjust_weights()
        6. Update team dynamic factors
        7. Generate and return report

        Args:
            match_results: Optional pre-loaded results. If None, attempts to
                fetch from data source with 30s timeout.

        Returns:
            RecalibrationReport with before/after weights and accuracy metrics.
        """
        # Step 1: Get match results
        if match_results is None:
            match_results = await self._fetch_results_with_timeout()

        if not match_results:
            return RecalibrationReport(
                matches_evaluated=0,
                weights_before=self._weights_to_dict(self.ensemble.weights),
                weights_after=self._weights_to_dict(self.ensemble.weights),
                exact_score_hits=0,
                direction_hits=0,
                avg_goal_error=0.0,
                systematic_bias=None,
            )

        # Step 2: Load predictions log
        predictions_log = self._load_predictions_log()

        # Step 3: Compare predictions vs actual
        comparisons = self._compare_predictions(match_results, predictions_log)

        # Step 4: Record weights before adjustment
        weights_before = self._weights_to_dict(self.ensemble.weights)

        # Step 5: Calculate performance and adjust weights
        if comparisons:
            performance = self._calculate_model_performance(comparisons)
            new_weights = self._adjust_weights(self.ensemble.weights, performance)
            self.ensemble.update_weights(new_weights)

        weights_after = self._weights_to_dict(self.ensemble.weights)

        # Step 6: Update dynamic factors for involved teams
        teams = self.data_manager.load_teams()
        teams_updated = self._update_dynamic_factors(teams, match_results)
        self._save_updated_teams(teams_updated)

        # Step 7: Generate report
        exact_hits = sum(1 for c in comparisons if c.exact_score_hit)
        direction_hits = sum(1 for c in comparisons if c.direction_hit)
        avg_error = (
            sum(c.goal_error for c in comparisons) / len(comparisons)
            if comparisons
            else 0.0
        )

        # Check for systematic bias (Requirement 4.8)
        systematic_bias = None
        if len(comparisons) >= self.BIAS_THRESHOLD_MATCHES:
            direction_rate = direction_hits / len(comparisons)
            if direction_rate < self.BIAS_DIRECTION_THRESHOLD:
                systematic_bias = self._generate_bias_report(
                    comparisons, match_results, teams_updated
                )

        # Save calibration history
        self._save_calibration_history(
            weights_before, weights_after, len(comparisons)
        )

        return RecalibrationReport(
            matches_evaluated=len(comparisons),
            weights_before=weights_before,
            weights_after=weights_after,
            exact_score_hits=exact_hits,
            direction_hits=direction_hits,
            avg_goal_error=round(avg_error, 2),
            systematic_bias=systematic_bias,
        )

    def _adjust_weights(
        self,
        current: EnsembleWeights,
        performance: dict[str, float],
    ) -> EnsembleWeights:
        """Adjust weights based on sub-model performance.

        Rules:
        - Each weight adjustment capped at ±MAX_WEIGHT_ADJUSTMENT (0.05)
        - All weights remain in [WEIGHT_MIN, WEIGHT_MAX] (i.e., [0.10, 0.60])
        - All weights sum to 1.00 after adjustment
        - Better performing models get weight increase, worse get decrease

        Args:
            current: Current ensemble weights.
            performance: Dict mapping model name -> performance score.
                Higher scores mean better performance.

        Returns:
            New adjusted EnsembleWeights.
        """
        weight_map = {
            "poisson": current.poisson,
            "elo": current.elo,
            "h2h": current.h2h,
            "dynamic": current.dynamic,
        }

        # Calculate average performance
        if not performance:
            return current

        avg_performance = sum(performance.values()) / len(performance)

        # Calculate raw adjustments based on deviation from average
        raw_adjustments: dict[str, float] = {}
        for model_name in weight_map:
            model_perf = performance.get(model_name, avg_performance)
            deviation = model_perf - avg_performance
            # Scale deviation to a reasonable adjustment range
            raw_adj = deviation * 0.1
            raw_adjustments[model_name] = raw_adj

        # Apply constraints: cap at ±MAX_WEIGHT_ADJUSTMENT
        capped_adjustments: dict[str, float] = {}
        for model_name, adj in raw_adjustments.items():
            capped = max(-self.MAX_WEIGHT_ADJUSTMENT, min(self.MAX_WEIGHT_ADJUSTMENT, adj))
            capped_adjustments[model_name] = capped

        # Apply adjustments and clamp to [WEIGHT_MIN, WEIGHT_MAX]
        new_weights: dict[str, float] = {}
        for model_name, weight in weight_map.items():
            adjusted = weight + capped_adjustments[model_name]
            clamped = max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, adjusted))
            new_weights[model_name] = clamped

        # Normalize to ensure sum = 1.00, while preserving ±0.05 max change
        new_weights = self._normalize_weights(new_weights, weight_map)

        return EnsembleWeights(
            poisson=new_weights["poisson"],
            elo=new_weights["elo"],
            h2h=new_weights["h2h"],
            dynamic=new_weights["dynamic"],
        )

    def _normalize_weights(
        self,
        weights: dict[str, float],
        original: dict[str, float],
    ) -> dict[str, float]:
        """Normalize weights to sum to 1.00 while respecting all constraints.

        Constraints:
        - Each weight in [WEIGHT_MIN, WEIGHT_MAX]
        - Each weight change from original is at most ±MAX_WEIGHT_ADJUSTMENT
        - Sum = 1.00

        Uses iterative adjustment to distribute the residual among weights
        that have room to adjust without violating any constraint.
        """
        for _ in range(20):  # Max iterations to converge
            total = sum(weights.values())
            if abs(total - 1.0) < 1e-9:
                break

            diff = 1.0 - total  # positive means we need to increase, negative decrease

            # Find weights that can still absorb adjustment in the needed direction
            adjustable = []
            for name, w in weights.items():
                orig = original[name]
                if diff > 0:
                    # Need to increase: check upper bounds
                    max_increase = min(
                        self.WEIGHT_MAX - w,
                        self.MAX_WEIGHT_ADJUSTMENT - (w - orig),
                    )
                    if max_increase > 1e-12:
                        adjustable.append((name, max_increase))
                else:
                    # Need to decrease: check lower bounds
                    max_decrease = min(
                        w - self.WEIGHT_MIN,
                        self.MAX_WEIGHT_ADJUSTMENT - (orig - w),
                    )
                    if max_decrease > 1e-12:
                        adjustable.append((name, max_decrease))

            if not adjustable:
                break

            # Distribute diff proportionally to available room
            total_room = sum(room for _, room in adjustable)
            for name, room in adjustable:
                share = (room / total_room) * diff
                # Apply while respecting per-item bounds
                if diff > 0:
                    actual = min(share, room)
                else:
                    actual = max(share, -room)
                weights[name] += actual

            # Clamp all weights to ensure bounds
            for name in weights:
                orig = original[name]
                weights[name] = max(self.WEIGHT_MIN, min(self.WEIGHT_MAX, weights[name]))
                # Also enforce ±0.05 from original
                weights[name] = max(
                    orig - self.MAX_WEIGHT_ADJUSTMENT,
                    min(orig + self.MAX_WEIGHT_ADJUSTMENT, weights[name]),
                )

        return weights

    # ========================================================================
    # DYNAMIC FACTOR UPDATES
    # ========================================================================

    def _update_dynamic_factors(
        self,
        teams: list[TeamProfile],
        results: list[MatchResult],
    ) -> list[TeamProfile]:
        """Update team dynamic factors after match results.

        For each result:
        - Winner: win_streak += 1, loss_streak = 0
        - Loser: loss_streak += 1, win_streak = 0
        - Draw: both streaks reset to 0
        - Update last_match_date to match date

        Args:
            teams: Current team profiles.
            results: New match results to process.

        Returns:
            Updated list of team profiles.
        """
        # Build team lookup by name
        team_map: dict[str, TeamProfile] = {t.name: t for t in teams}

        for result in results:
            team_a = team_map.get(result.team_a)
            team_b = team_map.get(result.team_b)

            if team_a:
                self._update_team_streak(team_a, result.score_a, result.score_b, result.date)
            if team_b:
                self._update_team_streak(team_b, result.score_b, result.score_a, result.date)

        return list(team_map.values())

    def _update_team_streak(
        self,
        team: TeamProfile,
        goals_for: int,
        goals_against: int,
        match_date: str,
    ) -> None:
        """Update a single team's streak and last match date.

        Args:
            team: The team profile to update (mutated in place).
            goals_for: Goals scored by this team.
            goals_against: Goals conceded by this team.
            match_date: ISO 8601 date of the match.
        """
        if goals_for > goals_against:
            # Team won
            team.current_win_streak += 1
            team.current_loss_streak = 0
        elif goals_for < goals_against:
            # Team lost
            team.current_loss_streak += 1
            team.current_win_streak = 0
        else:
            # Draw: both streaks reset
            team.current_win_streak = 0
            team.current_loss_streak = 0

        # Update last match date
        team.last_match_date = match_date

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def _fetch_results_with_timeout(self) -> list[MatchResult]:
        """Attempt to fetch results with 30s timeout.

        Falls back to loading from local data if timeout occurs.
        (Requirement 4.1, 4.2)
        """
        try:
            results = await asyncio.wait_for(
                self._fetch_from_data_source(),
                timeout=self.TIMEOUT_SECONDS,
            )
            return results
        except (asyncio.TimeoutError, Exception):
            # Fallback: load from local data (manual input scenario)
            return self.data_manager.load_match_results()

    async def _fetch_from_data_source(self) -> list[MatchResult]:
        """Fetch latest results from external data source.

        In practice, this would connect to a live API.
        For now, loads from local match_results.json.
        """
        # Simulate async data fetch - in production this would be an HTTP call
        await asyncio.sleep(0)  # Yield control
        return self.data_manager.load_match_results()

    def _load_predictions_log(self) -> list[dict]:
        """Load the predictions log from file."""
        filepath = self.data_manager.data_dir / "predictions_log.json"
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("predictions", [])

    def _compare_predictions(
        self,
        results: list[MatchResult],
        predictions: list[dict],
    ) -> list[MatchComparison]:
        """Compare predictions with actual results.

        Matches predictions to results by match_id.
        Calculates exact score hits, direction hits, and goal errors.
        """
        # Build prediction lookup by match_id
        pred_map: dict[str, dict] = {}
        for pred in predictions:
            pred_map[pred.get("match_id", "")] = pred

        comparisons: list[MatchComparison] = []

        for result in results:
            pred = pred_map.get(result.match_id)
            if not pred:
                continue

            predicted_score = tuple(pred.get("predicted_score", [0, 0]))
            actual_score = (result.score_a, result.score_b)

            predicted_direction = self._get_direction(predicted_score[0], predicted_score[1])
            actual_direction = self._get_direction(result.score_a, result.score_b)

            exact_hit = predicted_score == actual_score
            direction_hit = predicted_direction == actual_direction

            predicted_total = predicted_score[0] + predicted_score[1]
            actual_total = result.score_a + result.score_b
            goal_error = abs(predicted_total - actual_total)

            comparisons.append(
                MatchComparison(
                    match_id=result.match_id,
                    predicted_score=predicted_score,
                    actual_score=actual_score,
                    predicted_direction=predicted_direction,
                    actual_direction=actual_direction,
                    exact_score_hit=exact_hit,
                    direction_hit=direction_hit,
                    goal_error=goal_error,
                )
            )

        return comparisons

    def _calculate_model_performance(
        self, comparisons: list[MatchComparison]
    ) -> dict[str, float]:
        """Calculate performance score for each sub-model.

        Uses direction accuracy as primary metric.
        Models are scored based on how well the overall system performed,
        with adjustments based on which aspects each model contributes to:
        - Poisson: score accuracy (exact score hits + goal error)
        - Elo: direction accuracy
        - H2H: direction accuracy for repeat matchups
        - Dynamic: direction accuracy (captures momentum effects)

        Returns:
            Dict mapping model name to performance score (0.0 to 1.0).
        """
        if not comparisons:
            return {"poisson": 0.5, "elo": 0.5, "h2h": 0.5, "dynamic": 0.5}

        n = len(comparisons)
        direction_rate = sum(1 for c in comparisons if c.direction_hit) / n
        exact_rate = sum(1 for c in comparisons if c.exact_score_hit) / n
        avg_error = sum(c.goal_error for c in comparisons) / n

        # Score normalization: lower error = higher score
        error_score = max(0.0, 1.0 - avg_error / 4.0)  # 4 goals error → 0 score

        # Each model gets a composite score
        # Poisson is primarily about score prediction
        poisson_score = exact_rate * 0.6 + error_score * 0.4

        # Elo is about direction (win/lose/draw)
        elo_score = direction_rate

        # H2H is also about direction but with less data usually
        h2h_score = direction_rate * 0.8 + 0.1  # Slight baseline

        # Dynamic factor captures momentum, measured by direction
        dynamic_score = direction_rate * 0.7 + error_score * 0.3

        return {
            "poisson": poisson_score,
            "elo": elo_score,
            "h2h": h2h_score,
            "dynamic": dynamic_score,
        }

    def _generate_bias_report(
        self,
        comparisons: list[MatchComparison],
        results: list[MatchResult],
        teams: list[TeamProfile],
    ) -> str:
        """Generate cross-confederation bias report.

        Called when direction accuracy < 50% after 5+ matches.
        Reports which confederation pairings have poorest prediction accuracy.
        (Requirement 4.8)
        """
        team_map = {t.name: t for t in teams}

        # Group comparisons by confederation pairing
        confed_stats: dict[str, dict[str, int]] = {}  # "A vs B" -> {hits, total}

        for comp, result in zip(comparisons, results):
            team_a = team_map.get(result.team_a)
            team_b = team_map.get(result.team_b)
            if not team_a or not team_b:
                continue

            pairing = f"{team_a.confederation} vs {team_b.confederation}"
            if pairing not in confed_stats:
                confed_stats[pairing] = {"hits": 0, "total": 0}
            confed_stats[pairing]["total"] += 1
            if comp.direction_hit:
                confed_stats[pairing]["hits"] += 1

        # Build report
        lines = ["跨聯盟比賽系統性偏差分析報告:"]
        for pairing, stats in sorted(confed_stats.items()):
            rate = stats["hits"] / stats["total"] if stats["total"] > 0 else 0
            direction = "偏向高估" if rate < 0.5 else "準確"
            lines.append(
                f"  {pairing}: 命中率 {rate:.1%} ({stats['hits']}/{stats['total']}) - {direction}"
            )

        return "\n".join(lines)

    def _save_calibration_history(
        self,
        weights_before: dict[str, float],
        weights_after: dict[str, float],
        matches_evaluated: int,
    ) -> None:
        """Save calibration event to calibration.json."""
        filepath = self.data_manager.data_dir / "calibration.json"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {
                "current_weights": weights_after,
                "weight_history": [],
                "accuracy_records": [],
            }

        data["current_weights"] = weights_after
        data["weight_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "weights_before": weights_before,
            "weights_after": weights_after,
            "trigger": "auto_recalibration",
            "matches_evaluated": matches_evaluated,
        })

        self.data_manager._atomic_write(filepath, data)

    def _save_updated_teams(self, teams: list[TeamProfile]) -> None:
        """Save updated team profiles back to teams.json."""
        filepath = self.data_manager.data_dir / "teams.json"

        teams_data = []
        for team in teams:
            teams_data.append({
                "name": team.name,
                "name_zh": team.name_zh,
                "aliases": team.aliases,
                "confederation": team.confederation,
                "fifa_ranking": team.fifa_ranking,
                "fifa_points": team.fifa_points,
                "elo_rating": team.elo_rating,
                "group": team.group,
                "recent_goals_avg": team.recent_goals_avg,
                "recent_conceded_avg": team.recent_conceded_avg,
                "recent_win_rate": team.recent_win_rate,
                "recent_draw_rate": team.recent_draw_rate,
                "recent_loss_rate": team.recent_loss_rate,
                "neutral_win_rate": team.neutral_win_rate,
                "best_wc_result": team.best_wc_result,
                "vs_top20_win_rate": team.vs_top20_win_rate,
                "wc_first_match_win_rate": team.wc_first_match_win_rate,
                "penalty_shootout_win_rate": team.penalty_shootout_win_rate,
                "first_half_goal_pct": team.first_half_goal_pct,
                "second_half_goal_pct": team.second_half_goal_pct,
                "clean_sheet_rate": team.clean_sheet_rate,
                "failed_to_score_rate": team.failed_to_score_rate,
                "current_win_streak": team.current_win_streak,
                "current_loss_streak": team.current_loss_streak,
                "last_match_date": team.last_match_date,
                "eliminated_by_2022": team.eliminated_by_2022,
            })

        self.data_manager._atomic_write(filepath, {"teams": teams_data})

    @staticmethod
    def _get_direction(score_a: int, score_b: int) -> str:
        """Determine match direction from scores."""
        if score_a > score_b:
            return "win_a"
        elif score_a < score_b:
            return "win_b"
        return "draw"

    @staticmethod
    def _weights_to_dict(weights: EnsembleWeights) -> dict[str, float]:
        """Convert EnsembleWeights to plain dict."""
        return {
            "poisson": weights.poisson,
            "elo": weights.elo,
            "h2h": weights.h2h,
            "dynamic": weights.dynamic,
        }


# ============================================================================
# MCP TOOL HANDLER
# ============================================================================


async def handle_update_results(
    match_id: Optional[str],
    manual_result: Optional[str],
    data_manager: Optional[DataManager],
    ensemble: Optional[EnsembleModel],
    formatter=None,
) -> str:
    """Handle the update_results MCP tool invocation.

    Args:
        match_id: Optional specific match ID to update.
        manual_result: Optional manual result string.
        data_manager: Data manager instance.
        ensemble: Ensemble model instance.
        formatter: Output formatter instance.

    Returns:
        Recalibration report string or error message.
    """
    if data_manager is None or ensemble is None:
        return "Error: Server not initialized. Please restart the server."

    process = RecalibrationProcess(
        data_manager=data_manager,
        ensemble=ensemble,
    )

    # Parse manual result if provided
    manual_results: Optional[list[MatchResult]] = None
    if manual_result:
        parsed = _parse_manual_result(manual_result, match_id)
        if parsed:
            manual_results = [parsed]
            # Save to data store
            data_manager.save_match_result(parsed)

    # Execute recalibration
    report = await process.update_results(match_results=manual_results)

    # Format report
    lines = ["## 模型重新校準報告\n"]
    lines.append(f"評估比賽數: {report.matches_evaluated}")
    lines.append(f"精確比分命中: {report.exact_score_hits}")
    lines.append(f"勝負方向命中: {report.direction_hits}")
    lines.append(f"平均進球誤差: {report.avg_goal_error:.2f}")
    lines.append("\n### 權重變化")
    lines.append("| 模型 | 調整前 | 調整後 |")
    lines.append("|------|--------|--------|")
    for model in ["poisson", "elo", "h2h", "dynamic"]:
        before = report.weights_before.get(model, 0)
        after = report.weights_after.get(model, 0)
        lines.append(f"| {model} | {before:.3f} | {after:.3f} |")

    if report.systematic_bias:
        lines.append(f"\n### 系統性偏差分析\n{report.systematic_bias}")

    return "\n".join(lines)


def _parse_manual_result(
    manual_result: str, match_id: Optional[str] = None
) -> Optional[MatchResult]:
    """Parse a manual result string into a MatchResult.

    Expected format: "TeamA ScoreA - ScoreB TeamB"
    Example: "Brazil 2 - 1 Argentina"

    Returns None if parsing fails.
    """
    try:
        parts = manual_result.split("-")
        if len(parts) != 2:
            return None

        left = parts[0].strip().rsplit(" ", 1)
        right = parts[1].strip().split(" ", 1)

        if len(left) != 2 or len(right) != 2:
            return None

        team_a = left[0].strip()
        score_a = int(left[1].strip())
        score_b = int(right[0].strip())
        team_b = right[1].strip()

        return MatchResult(
            match_id=match_id or f"MANUAL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            team_a=team_a,
            team_b=team_b,
            score_a=score_a,
            score_b=score_b,
            stage="group",
        )
    except (ValueError, IndexError):
        return None
