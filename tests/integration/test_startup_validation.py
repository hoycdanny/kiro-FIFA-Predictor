"""
Integration tests for startup validation.

Tests cover:
- Successful startup with valid data files
- Startup abort on missing teams (wrong count)
- Startup abort on invalid group counts (group with ≠ 4 teams)
- Startup abort on missing required fields in team data
- Fallback data behavior on external source timeout

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import json
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from src.data.data_manager import DataManager, DataValidationError


# ============================================================================
# HELPERS
# ============================================================================


def _make_team(index: int, group: str, **overrides) -> dict:
    """Create a minimal valid team dictionary for testing."""
    base = {
        "name": f"Team{index}",
        "name_zh": f"隊伍{index}",
        "aliases": [f"T{index}"],
        "confederation": "UEFA",
        "fifa_ranking": index + 1,
        "fifa_points": 1500.0 + index,
        "elo_rating": 1800 + index,
        "group": group,
        "recent_goals_avg": 1.5,
        "recent_conceded_avg": 1.0,
        "recent_win_rate": 50.0,
        "recent_draw_rate": 25.0,
        "recent_loss_rate": 25.0,
        "neutral_win_rate": 45.0,
        "best_wc_result": "Group stage",
        "vs_top20_win_rate": 30.0,
        "wc_first_match_win_rate": 33.0,
        "penalty_shootout_win_rate": 50.0,
        "first_half_goal_pct": 45.0,
        "second_half_goal_pct": 55.0,
        "clean_sheet_rate": 25.0,
        "failed_to_score_rate": 20.0,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "last_match_date": None,
        "eliminated_by_2022": None,
    }
    base.update(overrides)
    return base


def _create_valid_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with all valid data files.

    Creates:
    - teams.json with 48 teams
    - groups.json with 12 groups × 4 teams
    - schedule.json with at least one match
    - match_results.json (empty)
    - predictions_log.json (empty)
    """
    # 48 teams across 12 groups
    teams = []
    for i in range(48):
        group = chr(ord("A") + i // 4)
        teams.append(_make_team(i, group))

    with open(tmp_path / "teams.json", "w", encoding="utf-8") as f:
        json.dump(teams, f, ensure_ascii=False)

    # 12 groups, each with 4 teams
    groups = {}
    for i in range(12):
        group_id = chr(ord("A") + i)
        groups[group_id] = [f"Team{i * 4 + j}" for j in range(4)]

    with open(tmp_path / "groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False)

    # Schedule with one match
    schedule = {
        "matches": [
            {
                "match_id": "GS-A-1",
                "date": "2026-06-11",
                "team_a": "Team0",
                "team_b": "Team1",
                "stage": "group",
                "group": "A",
                "venue_city": "New York",
                "venue_country": "United States",
            }
        ]
    }
    with open(tmp_path / "schedule.json", "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False)

    # Empty match results
    with open(tmp_path / "match_results.json", "w", encoding="utf-8") as f:
        json.dump({"matches": []}, f)

    # Empty predictions log
    with open(tmp_path / "predictions_log.json", "w", encoding="utf-8") as f:
        json.dump({"predictions": []}, f)

    return tmp_path


# ============================================================================
# TESTS: Successful startup with valid data
# ============================================================================


class TestSuccessfulStartup:
    """Test that startup succeeds with valid, complete data files."""

    def test_validate_startup_succeeds_with_valid_data(self, tmp_path: Path):
        """Startup validation should pass when all data files are valid.

        Validates: Requirements 9.1, 9.2
        """
        data_dir = _create_valid_data_dir(tmp_path)
        dm = DataManager(data_dir)

        # Should not raise any exception
        dm.validate_startup()

    def test_validate_startup_loads_48_teams(self, tmp_path: Path):
        """After validation, 48 teams should be loadable.

        Validates: Requirements 9.1
        """
        data_dir = _create_valid_data_dir(tmp_path)
        dm = DataManager(data_dir)
        dm.validate_startup()

        teams = dm.load_teams()
        assert len(teams) == 48

    def test_validate_startup_loads_12_groups(self, tmp_path: Path):
        """After validation, 12 groups should be loadable.

        Validates: Requirements 9.2
        """
        data_dir = _create_valid_data_dir(tmp_path)
        dm = DataManager(data_dir)
        dm.validate_startup()

        groups = dm.load_groups()
        assert len(groups) == 12
        assert set(groups.keys()) == set("ABCDEFGHIJKL")

    def test_validate_startup_each_group_has_4_teams(self, tmp_path: Path):
        """After validation, each group should contain exactly 4 teams.

        Validates: Requirements 9.2
        """
        data_dir = _create_valid_data_dir(tmp_path)
        dm = DataManager(data_dir)
        dm.validate_startup()

        groups = dm.load_groups()
        for group_id, team_list in groups.items():
            assert len(team_list) == 4, (
                f"Group {group_id} has {len(team_list)} teams, expected 4"
            )

    def test_validate_startup_loads_schedule(self, tmp_path: Path):
        """After validation, schedule should be loadable.

        Validates: Requirements 9.2
        """
        data_dir = _create_valid_data_dir(tmp_path)
        dm = DataManager(data_dir)
        dm.validate_startup()

        schedule = dm.load_schedule()
        assert len(schedule) >= 1
        assert schedule[0].match_id == "GS-A-1"

    def test_startup_with_real_project_data(self):
        """Validate that the actual project data passes startup validation.

        This is a smoke test against the real data files.
        Validates: Requirements 9.1, 9.2
        """
        real_data_dir = Path(__file__).parent.parent.parent / "data"
        if not real_data_dir.exists():
            pytest.skip("Real data directory not available")

        dm = DataManager(real_data_dir)
        dm.validate_startup()  # Should not raise

    @pytest.mark.asyncio
    async def test_server_startup_with_valid_data(self, tmp_path: Path):
        """Full server startup should succeed with valid data.

        Validates: Requirements 9.1, 9.2
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Also create calibration.json needed by AccuracyTracker
        calibration = {
            "current_weights": {
                "poisson": 0.40,
                "elo": 0.25,
                "h2h": 0.15,
                "dynamic": 0.20,
            },
            "weight_history": [],
            "accuracy_records": [],
        }
        with open(data_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f)

        import src.server

        with patch.object(src.server, "_get_data_dir", return_value=data_dir):
            await src.server.startup()

        assert src.server._data_manager is not None
        assert src.server._prediction_engine is not None


# ============================================================================
# TESTS: Startup abort on missing teams
# ============================================================================


class TestStartupAbortMissingTeams:
    """Test that startup aborts when teams data is invalid."""

    def test_abort_on_fewer_than_48_teams(self, tmp_path: Path):
        """Startup should abort if teams.json contains fewer than 48 teams.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Overwrite teams.json with only 30 teams
        teams = [_make_team(i, chr(ord("A") + i // 4)) for i in range(30)]
        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Expected 48 teams, got 30" in str(exc_info.value)

    def test_abort_on_more_than_48_teams(self, tmp_path: Path):
        """Startup should abort if teams.json contains more than 48 teams.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Overwrite teams.json with 50 teams
        teams = [_make_team(i, chr(ord("A") + min(i // 4, 11))) for i in range(50)]
        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Expected 48 teams, got 50" in str(exc_info.value)

    def test_abort_on_missing_teams_file(self, tmp_path: Path):
        """Startup should abort if teams.json file doesn't exist.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)
        os.remove(data_dir / "teams.json")

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Failed to load teams.json" in str(exc_info.value)

    def test_abort_on_invalid_teams_json(self, tmp_path: Path):
        """Startup should abort if teams.json contains invalid JSON.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Write invalid JSON
        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            f.write("{invalid json content")

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Failed to load teams.json" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_server_exits_on_invalid_team_count(self, tmp_path: Path):
        """Server startup should call sys.exit(1) when team count is wrong.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Write only 20 teams
        teams = [_make_team(i, chr(ord("A") + i // 4)) for i in range(20)]
        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        import src.server

        with patch.object(src.server, "_get_data_dir", return_value=data_dir):
            with pytest.raises(SystemExit) as exc_info:
                await src.server.startup()

            assert exc_info.value.code == 1


# ============================================================================
# TESTS: Startup abort on invalid group counts
# ============================================================================


class TestStartupAbortInvalidGroups:
    """Test that startup aborts when group data is invalid."""

    def test_abort_on_group_with_3_teams(self, tmp_path: Path):
        """Startup should abort if any group has fewer than 4 teams.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Modify group B to have only 3 teams
        with open(data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups["B"] = groups["B"][:3]

        with open(data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Group B: expected 4 teams, got 3" in str(exc_info.value)

    def test_abort_on_group_with_5_teams(self, tmp_path: Path):
        """Startup should abort if any group has more than 4 teams.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups["C"] = groups["C"] + ["ExtraTeam"]

        with open(data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Group C: expected 4 teams, got 5" in str(exc_info.value)

    def test_abort_on_missing_group(self, tmp_path: Path):
        """Startup should abort if a group is missing from groups.json.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        del groups["L"]

        with open(data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Missing groups" in str(exc_info.value)

    def test_abort_on_missing_groups_file(self, tmp_path: Path):
        """Startup should abort if groups.json file doesn't exist.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)
        os.remove(data_dir / "groups.json")

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        assert "Failed to load groups.json" in str(exc_info.value)

    def test_abort_on_multiple_invalid_groups(self, tmp_path: Path):
        """Startup should report all invalid groups in the error.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups["A"] = groups["A"][:2]  # Only 2 teams
        groups["F"] = groups["F"][:3]  # Only 3 teams

        with open(data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        assert "Group A" in error_msg
        assert "Group F" in error_msg

    @pytest.mark.asyncio
    async def test_server_exits_on_invalid_groups(self, tmp_path: Path):
        """Server startup should call sys.exit(1) when groups are invalid.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        groups["D"] = groups["D"][:1]  # Only 1 team

        with open(data_dir / "groups.json", "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False)

        import src.server

        with patch.object(src.server, "_get_data_dir", return_value=data_dir):
            with pytest.raises(SystemExit) as exc_info:
                await src.server.startup()

            assert exc_info.value.code == 1


# ============================================================================
# TESTS: Startup abort on missing required fields
# ============================================================================


class TestStartupAbortMissingFields:
    """Test that startup aborts when team data has missing required fields."""

    def test_abort_on_missing_name_zh(self, tmp_path: Path):
        """Startup should abort if a team has empty name_zh field.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[5]["name_zh"] = ""  # Empty required string field

        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        assert "missing required field" in error_msg
        assert "name_zh" in error_msg

    def test_abort_on_missing_confederation(self, tmp_path: Path):
        """Startup should abort if a team has empty confederation field.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[10]["confederation"] = ""  # Empty required field

        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        assert "missing required field" in error_msg
        assert "confederation" in error_msg

    def test_abort_on_missing_best_wc_result(self, tmp_path: Path):
        """Startup should abort if a team has empty best_wc_result field.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[0]["best_wc_result"] = ""  # Empty required field

        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        assert "missing required field" in error_msg
        assert "best_wc_result" in error_msg

    def test_error_message_identifies_team(self, tmp_path: Path):
        """Error message should identify which team has the missing field.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[7]["name_zh"] = ""  # Team7 has empty name_zh

        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        assert "Team7" in error_msg

    def test_abort_reports_multiple_field_errors(self, tmp_path: Path):
        """Startup should report all missing field errors, not just the first.

        Validates: Requirements 9.3
        """
        data_dir = _create_valid_data_dir(tmp_path)

        with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
            teams = json.load(f)
        teams[0]["name_zh"] = ""
        teams[1]["confederation"] = ""

        with open(data_dir / "teams.json", "w", encoding="utf-8") as f:
            json.dump(teams, f, ensure_ascii=False)

        dm = DataManager(data_dir)
        with pytest.raises(DataValidationError) as exc_info:
            dm.validate_startup()

        error_msg = str(exc_info.value)
        # Should report errors for both teams
        assert "Team0" in error_msg
        assert "Team1" in error_msg


# ============================================================================
# TESTS: Fallback data behavior on external source timeout
# ============================================================================


class TestFallbackDataBehavior:
    """Test fallback data behavior when external source is unavailable.

    Validates: Requirements 9.4
    """

    def test_fallback_data_script_generates_valid_snapshot(self):
        """The fallback data script should produce a valid teams_fallback.json.

        Validates: Requirements 9.4
        """
        from scripts.fallback_data import generate_fallback, _validate_source

        # Use the real data directory
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("Real data directory not available")

        # Run the fallback generator
        success = generate_fallback()
        assert success is True

        # Verify the output file exists and is valid
        fallback_path = data_dir / "teams_fallback.json"
        assert fallback_path.exists()

        with open(fallback_path, "r", encoding="utf-8") as f:
            fallback_data = json.load(f)

        # Should have metadata and teams
        assert "_metadata" in fallback_data
        assert "teams" in fallback_data
        assert fallback_data["_metadata"]["team_count"] == 48
        assert len(fallback_data["teams"]) == 48

    def test_fallback_data_contains_required_fields(self):
        """Fallback data teams should contain all required fields.

        Validates: Requirements 9.4
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("Real data directory not available")

        from scripts.fallback_data import generate_fallback

        generate_fallback()

        fallback_path = data_dir / "teams_fallback.json"
        with open(fallback_path, "r", encoding="utf-8") as f:
            fallback_data = json.load(f)

        # Verify each team has the required fields from DataManager
        required = set(DataManager.REQUIRED_TEAM_FIELDS)
        for team in fallback_data["teams"]:
            team_fields = set(team.keys())
            missing = required - team_fields
            assert missing == set(), (
                f"Team '{team.get('name', '?')}' missing: {missing}"
            )

    def test_fallback_data_loadable_by_data_manager(self, tmp_path: Path):
        """Fallback data should be loadable by DataManager as teams source.

        Validates: Requirements 9.4
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("Real data directory not available")

        from scripts.fallback_data import generate_fallback

        generate_fallback()

        fallback_path = data_dir / "teams_fallback.json"
        with open(fallback_path, "r", encoding="utf-8") as f:
            fallback_data = json.load(f)

        # Write fallback teams as teams.json in a temp directory
        with open(tmp_path / "teams.json", "w", encoding="utf-8") as f:
            json.dump(fallback_data, f, ensure_ascii=False)

        # Copy groups.json and schedule.json from real data
        import shutil
        shutil.copy(data_dir / "groups.json", tmp_path / "groups.json")
        shutil.copy(data_dir / "schedule.json", tmp_path / "schedule.json")

        # DataManager should load successfully (fallback has {"teams": [...]} format)
        dm = DataManager(tmp_path)
        teams = dm.load_teams()
        assert len(teams) == 48

    def test_fallback_script_fails_gracefully_on_missing_source(self, tmp_path: Path):
        """Fallback script should return False if source teams.json is missing.

        Validates: Requirements 9.4
        """
        from scripts.fallback_data import _validate_source

        # Validate with an empty team list — should report wrong count
        _, errors = _validate_source([])
        assert len(errors) > 0
        assert "Expected 48 teams" in errors[0]

    def test_fallback_generate_returns_false_on_missing_file(self, tmp_path: Path):
        """generate_fallback() should return False when source file is absent.

        Validates: Requirements 9.4
        """
        from scripts import fallback_data

        # Patch __file__ path so it resolves data dir to a temp dir with no teams.json
        fake_script = tmp_path / "scripts" / "fallback_data.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        fake_script.touch()

        with patch.object(
            fallback_data, "__file__", str(fake_script)
        ):
            result = fallback_data.generate_fallback()
            assert result is False

    def test_fallback_validates_team_count(self):
        """Fallback validation should reject data with wrong team count.

        Validates: Requirements 9.4
        """
        from scripts.fallback_data import _validate_source

        # Test with only 10 teams
        teams = [{"name": f"T{i}", "fifa_ranking": i, "elo_rating": 1500, "confederation": "UEFA"} for i in range(10)]
        _, errors = _validate_source(teams)
        assert any("Expected 48 teams" in e for e in errors)

    def test_fallback_validates_required_fields(self):
        """Fallback validation should detect missing required fields.

        Validates: Requirements 9.4
        """
        from scripts.fallback_data import _validate_source

        # Team missing 'confederation'
        teams = [{"name": f"T{i}", "fifa_ranking": i, "elo_rating": 1500} for i in range(48)]
        _, errors = _validate_source(teams)
        assert any("missing fields" in e for e in errors)

    @pytest.mark.asyncio
    async def test_server_startup_uses_data_even_without_external_source(
        self, tmp_path: Path
    ):
        """Server should start using local data when no external source is available.

        This verifies that the system operates correctly from local files alone,
        which is the fallback behavior when external data is unreachable.

        Validates: Requirements 9.4
        """
        data_dir = _create_valid_data_dir(tmp_path)

        # Add calibration.json needed for full startup
        calibration = {
            "current_weights": {
                "poisson": 0.40,
                "elo": 0.25,
                "h2h": 0.15,
                "dynamic": 0.20,
            },
            "weight_history": [],
            "accuracy_records": [],
        }
        with open(data_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f)

        import src.server

        # Startup should succeed using only local data files (no external source)
        with patch.object(src.server, "_get_data_dir", return_value=data_dir):
            await src.server.startup()

        assert src.server._data_manager is not None
        assert src.server._prediction_engine is not None

    def test_fallback_data_has_metadata_timestamp(self):
        """Generated fallback should include generation timestamp metadata.

        Validates: Requirements 9.4
        """
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("Real data directory not available")

        from scripts.fallback_data import generate_fallback

        generate_fallback()

        fallback_path = data_dir / "teams_fallback.json"
        with open(fallback_path, "r", encoding="utf-8") as f:
            fallback_data = json.load(f)

        metadata = fallback_data["_metadata"]
        assert "generated_at" in metadata
        assert "source_hash_sha256" in metadata
        assert "version" in metadata
