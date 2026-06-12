# Implementation Plan: FIFA Predictor Power

## Overview

Implement a Kiro Power providing 2026 FIFA World Cup match prediction capabilities via MCP tools. The system uses a multi-model ensemble prediction engine (Dixon-Coles Poisson, Elo, Historical H2H, Dynamic Factors) with coach style analysis, Monte Carlo tournament simulation, post-match recalibration, and accuracy tracking. Built with Python, FastMCP SDK, NumPy, and Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure, dependencies, and core data models
  - [x] 1.1 Initialize Python project with pyproject.toml, directory structure, and dependencies
    - Create `pyproject.toml` with dependencies: `mcp[cli]`, `numpy`, `rich`; dev deps: `pytest`, `hypothesis`, `pytest-asyncio`, `pytest-cov`
    - Create directory structure: `src/`, `src/engine/`, `src/data/`, `src/tools/`, `src/output/`, `src/utils/`, `data/`, `scripts/`, `tests/`, `tests/properties/`, `tests/unit/`, `tests/integration/`
    - Add `__init__.py` files to all Python package directories
    - Create `.gitignore` for Python project
    - _Requirements: 9.1, 9.2_

  - [x] 1.2 Implement core data models and constants
    - Create `src/utils/constants.py` with all 48 team names, aliases (English/Chinese/abbreviations), confederation mappings, host nations set, group assignments (A-L), and model default parameters
    - Create `TeamProfile` dataclass in `src/data/data_manager.py` with all 19+ fields as specified in design (fifa_ranking, elo_rating, recent_goals_avg, etc.)
    - Create supporting dataclasses: `MatchResult`, `ScheduleEntry`, `PredictionLogEntry`, `PredictionError`
    - _Requirements: 6.2, 9.1_

  - [x] 1.3 Create initial JSON data files for 48 teams, 12 groups, and schedule
    - Create `data/teams.json` with complete TeamProfile data for all 48 participating teams
    - Create `data/groups.json` with 12 groups (A-L), each containing exactly 4 teams
    - Create `data/schedule.json` with group stage match schedule
    - Create empty `data/match_results.json` (structure: `{"matches": []}`)
    - Create empty `data/predictions_log.json` (structure: `{"predictions": []}`)
    - Create `data/calibration.json` with default weights
    - _Requirements: 9.1, 9.2_

- [x] 2. Implement data layer and utilities
  - [x] 2.1 Implement DataManager with atomic write and data loading
    - Implement `DataManager` class in `src/data/data_manager.py` with `load_teams()`, `load_groups()`, `load_schedule()`
    - Implement `_atomic_write()` using tempfile + `os.replace()` pattern
    - Implement `save_match_result()` and `append_prediction_log()` with atomic writes
    - Implement startup validation: verify 48 teams, 12 groups × 4 teams, no missing required fields
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6_

  - [x] 2.2 Implement TeamMatcher for fuzzy name resolution
    - Create `src/utils/team_matcher.py` with `TeamMatcher` class
    - Implement fuzzy matching using `difflib.SequenceMatcher` supporting English names, Chinese names, abbreviations, and partial matches
    - Return exact match, single fuzzy match, multiple matches (for user selection), or no match (with up to 3 suggestions)
    - Handle case-insensitive matching
    - _Requirements: 1.5, 1.8, 6.1, 6.3, 6.4_

  - [x] 2.3 Implement InputValidator
    - Create `src/utils/validator.py` with `InputValidator` class
    - Implement `validate_team()`, `validate_group()`, `validate_coach_style()` methods
    - Return structured `PredictionError` with error_code, message, and suggestions on failure
    - Support group validation (A-L, case-insensitive), coach style validation (Chinese/English names and keywords)
    - _Requirements: 1.5, 1.7, 2.4, 6.3_

  - [x]* 2.4 Write property tests for data layer
    - **Property 17: Atomic write data integrity**
    - **Property 18: Prediction log entry completeness**
    - **Validates: Requirements 9.5, 9.6**

  - [x]* 2.5 Write property tests for team name resolution
    - **Property 4: Team name resolution bidirectional consistency**
    - **Property 5: Invalid team suggestion bounds**
    - **Validates: Requirements 1.5, 1.8, 6.1, 6.3**

- [x] 3. Checkpoint - Ensure data layer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement prediction sub-models
  - [x] 4.1 Implement Dixon-Coles Poisson Model
    - Create `src/engine/dixon_coles.py` with `DixonColesModel` class
    - Implement `predict()` returning 5×5 score probability matrix
    - Calculate `attack_strength = (team_goals_avg / league_avg) × confederation_coeff`
    - Calculate `defense_weakness = (opponent_conceded_avg / league_avg)`
    - Calculate `lambda = attack_strength × defense_weakness × neutral_factor`
    - Implement `_tau_correction()` for 0-0, 1-0, 0-1, 1-1 low-score adjustments
    - Implement `_poisson_probability()` using NumPy
    - _Requirements: 7.1_

  - [x] 4.2 Implement Elo Rating Model
    - Create `src/engine/elo_model.py` with `EloModel` class
    - Implement `predict()` using formula: `P(A) = 1 / (1 + 10^((Elo_B - Elo_A + home_advantage) / 400))`
    - Set `home_advantage = 0` for neutral venue
    - Apply `+50` Elo bonus for host nations (USA, Canada, Mexico) playing in their country
    - Derive draw probability from win probabilities
    - _Requirements: 7.2, 7.3_

  - [x] 4.3 Implement Historical H2H Model
    - Create `src/engine/h2h_model.py` with `H2HModel` class
    - Use historical head-to-head records to compute win/draw/lose probabilities
    - Handle cases where no H2H data exists (fall back to neutral 33/34/33 split)
    - _Requirements: 7.4_

  - [x] 4.4 Implement Dynamic Factor Model
    - Create `src/engine/dynamic_factor.py` with `DynamicFactorModel` and `DynamicFactors` dataclass
    - Implement `calculate_adjustment()`: +5% for win_streak ≥ 3, -5% for loss_streak ≥ 3, -3% for rest < 3 days, +3% for revenge factor (2022 eliminator)
    - Only applicable conditions contribute to adjustment
    - _Requirements: 7.5, 7.6, 7.7_

  - [x]* 4.5 Write property tests for prediction sub-models
    - **Property 11: Dixon-Coles score matrix validity**
    - **Property 12: Elo model probability completeness**
    - **Property 10: Dynamic factor composite calculation**
    - **Validates: Requirements 7.1, 7.2, 7.5, 7.6, 7.7**

- [x] 5. Implement ensemble model and coach style system
  - [x] 5.1 Implement Ensemble Model
    - Create `src/engine/ensemble.py` with `EnsembleModel` and `EnsembleWeights` classes
    - Implement weighted combination: Poisson 0.40, Elo 0.25, H2H 0.15, Dynamic 0.20
    - Implement `validate()`: each weight in [0.10, 0.60], sum = 1.00
    - Implement `redistribute_without()` for model exclusion (proportional redistribution)
    - Implement `predict_with_fallback()` to handle sub-model failures gracefully
    - _Requirements: 7.4, 7.8_

  - [x] 5.2 Implement Coach Style System
    - Create `src/engine/coach_style.py` with `CoachStyleSystem`, `CoachStyleType` enum, and keyword mappings
    - Implement `_apply_analyst()`: no modification, return original prediction
    - Implement `_apply_contrarian()`: boost underdog to 35-40% if below 35%, recommend upset scores
    - Implement `_apply_tactician()`: adjust based on streaks, revenge factor, fatigue
    - Implement `generate_narrative()` with fixed prefixes per style
    - Normalize probabilities after style adjustments to maintain W+D+L = 100%
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [x]* 5.3 Write property tests for ensemble and coach style
    - **Property 8: Ensemble weight invariant**
    - **Property 13: Model exclusion weight redistribution**
    - **Property 14: Analyst style identity**
    - **Property 15: Contrarian style underdog boost with probability preservation**
    - **Property 16: Coach style narrative prefix**
    - **Validates: Requirements 7.4, 7.8, 4.4, 8.1, 8.2, 8.7**

- [x] 6. Implement Prediction Engine and Monte Carlo Simulator
  - [x] 6.1 Implement Prediction Engine main controller
    - Create `src/engine/prediction_engine.py` with `PredictionEngine` class and `MatchPrediction` dataclass
    - Implement `predict_match()`: orchestrate ensemble prediction, apply coach style, compute confidence index, over/under 2.5, top 3 scores
    - Ensure W+D+L sums to 100.0% and over+under sums to 100.0%
    - Implement `predict_group()`: simulate all 6 group matches, rank by points > goal difference > goals for
    - Mark top 2 as "確定晉級", compute 3rd place qualification probability
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.5_

  - [x] 6.2 Implement Monte Carlo Simulator
    - Create `src/engine/monte_carlo.py` with `MonteCarloSimulator`, `TournamentResult`, and `ChampionPrediction` dataclasses
    - Implement `simulate_tournament()` with configurable simulation count (default 10,000)
    - Simulate full knockout bracket: round of 32, round of 16, quarter-finals, semi-finals, third place, final
    - Use NumPy vectorization for batch simulation performance
    - Calculate per-team round advancement probabilities and confidence index based on convergence
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 6.3 Write property tests for prediction engine and Monte Carlo
    - **Property 1: Win/Draw/Lose probability sum invariant**
    - **Property 2: Over/Under probability sum invariant**
    - **Property 3: Confidence index range invariant**
    - **Property 6: Group simulation structural invariants**
    - **Property 7: Tournament round probability consistency**
    - **Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.2, 2.5, 3.2**

- [x] 7. Checkpoint - Ensure core engine tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement recalibration and accuracy tracking
  - [x] 8.1 Implement Recalibration Process
    - Create `src/tools/update_results.py` with `RecalibrationProcess` class
    - Implement `update_results()`: fetch results (30s timeout), compare predictions vs actual, adjust weights, update dynamic factors
    - Implement `_adjust_weights()`: max ±0.05 per adjustment, weights stay in [0.10, 0.60], sum = 1.00
    - Update team dynamic factors (streaks, fatigue, revenge) after each result
    - Generate recalibration report with before/after weights and accuracy changes
    - Handle timeout with manual input fallback
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x] 8.2 Implement Accuracy Tracker
    - Create `src/utils/accuracy_tracker.py` with `AccuracyTracker` and `AccuracyReport` classes
    - Implement `calculate_report()`: exact score rate, direction rate, avg goal error
    - Calculate per-coach-style accuracy breakdown
    - Calculate confidence calibration (3 bands: 0-33, 34-66, 67-100)
    - Calculate cross-confederation accuracy analysis
    - Return "insufficient data" message if < 3 matches completed
    - Return "no data" message if zero matches have results
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x]* 8.3 Write property tests for recalibration and accuracy
    - **Property 9: Recalibration adjustment magnitude bound**
    - **Property 20: Streak counter update correctness**
    - **Property 19: Accuracy metric calculation correctness**
    - **Validates: Requirements 4.4, 4.5, 4.7, 5.1, 5.2, 5.3**

- [x] 9. Implement output formatting layer
  - [x] 9.1 Implement output formatter and Markdown renderer
    - Create `src/output/formatter.py` with `OutputFormatter` class that dispatches to appropriate renderer
    - Create `src/output/markdown_renderer.py` with `MarkdownRenderer` for Kiro chat interface output
    - Format single match prediction: team names with flag emoji, top 3 scores, W/D/L probabilities, confidence, xG, over/under
    - Format group standings as markdown table (Rank, Team, P, W, D, L, GF, GA, GD)
    - Format champion prediction as bracket tree structure
    - Format team profile in categorized sections (Basic Info, Recent Form, WC History, Advanced Stats)
    - Append `data_updated_at` (ISO 8601) and `model_version` to all outputs
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.7_

  - [x] 9.2 Implement Rich CLI renderer
    - Create `src/output/rich_renderer.py` with `RichRenderer` class
    - Use Rich library for tables with borders, color, and alignment
    - Support all output types: match prediction, group standings, champion bracket, team profile
    - _Requirements: 10.6_

- [x] 10. Implement MCP tools and server
  - [x] 10.1 Implement MCP server entry point
    - Create `src/server.py` with FastMCP server initialization
    - Implement `create_server()` and `startup()` functions
    - Register all 6 MCP tools via `@mcp.tool()` decorators
    - Wire up DataManager, PredictionEngine, and all dependencies at startup
    - Abort startup with specific error if data validation fails
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 10.2 Implement predict_match MCP tool
    - Create `src/tools/predict_match.py` with `predict_match` tool function
    - Accept parameters: `team_a` (str), `team_b` (str), `coach_style` (optional str)
    - Validate inputs, resolve team names, execute prediction, format output
    - Display all three coach styles in output (with default being analyst)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 8.4, 8.5_

  - [x] 10.3 Implement predict_group MCP tool
    - Create `src/tools/predict_group.py` with `predict_group` tool function
    - Accept parameter: `group_id` (str, A-L)
    - Validate group ID, simulate 6 matches, compute standings, format table output
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 10.4 Implement predict_champion MCP tool
    - Create `src/tools/predict_champion.py` with `predict_champion` tool function
    - Accept parameter: `simulations` (optional int, default 10000)
    - Run Monte Carlo simulation, format bracket and top-5 output
    - Handle case where group stage not complete (use group predictions as input)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 10.5 Implement update_results MCP tool
    - Create `src/tools/update_results.py` tool function wiring to `RecalibrationProcess`
    - Accept parameters: `match_id` (optional), `manual_result` (optional fallback input)
    - Execute recalibration flow and return report
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 10.6 Implement accuracy_stats MCP tool
    - Create `src/tools/accuracy_stats.py` with `accuracy_stats` tool function
    - No required parameters
    - Return accuracy report or insufficient data message
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 10.7 Implement team_info MCP tool
    - Create `src/tools/team_info.py` with `team_info` tool function
    - Accept parameter: `team_name` (str)
    - Resolve name via TeamMatcher, handle ambiguous matches, format team profile output
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 11. Implement Kiro Power configuration
  - [x] 11.1 Create POWER.md and config.json for Kiro Power registration
    - Create `.kiro/powers/fifa-predictor/POWER.md` with power documentation, tool descriptions, and usage examples
    - Create `.kiro/powers/fifa-predictor/config.json` with MCP server configuration (command: python, args, env)
    - _Requirements: 9.1_

  - [x] 11.2 Create fallback data script
    - Create `scripts/fallback_data.py` to generate static data snapshot for offline use
    - Script produces a frozen copy of teams.json usable when external data source is unavailable
    - _Requirements: 9.4_

- [x] 12. Checkpoint - Ensure all unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Write property-based tests for all correctness properties
  - [x]* 13.1 Write property tests for probability invariants (Properties 1-3)
    - **Property 1: Win/Draw/Lose probability sum invariant**
    - **Property 2: Over/Under probability sum invariant**
    - **Property 3: Confidence index range invariant**
    - Create `tests/properties/test_prop_probabilities.py`
    - Use Hypothesis strategies for team pairs and coach styles
    - **Validates: Requirements 1.2, 1.3, 1.4**

  - [x]* 13.2 Write property tests for team resolution (Properties 4-5)
    - **Property 4: Team name resolution bidirectional consistency**
    - **Property 5: Invalid team suggestion bounds**
    - Create `tests/properties/test_prop_team_matcher.py`
    - Use Hypothesis strategies for team names and random strings
    - **Validates: Requirements 1.5, 1.8, 6.1, 6.3**

  - [x]* 13.3 Write property tests for group and tournament simulation (Properties 6-7)
    - **Property 6: Group simulation structural invariants**
    - **Property 7: Tournament round probability consistency**
    - Create `tests/properties/test_prop_group_sim.py`
    - Use Hypothesis strategies for group IDs and qualified team lists
    - **Validates: Requirements 2.1, 2.2, 2.5, 3.2**

  - [x]* 13.4 Write property tests for ensemble model (Properties 8-9, 13)
    - **Property 8: Ensemble weight invariant**
    - **Property 9: Recalibration adjustment magnitude bound**
    - **Property 13: Model exclusion weight redistribution**
    - Create `tests/properties/test_prop_ensemble.py`
    - Use Hypothesis strategies for weight values and model names
    - **Validates: Requirements 7.4, 4.4, 4.7, 7.8**

  - [x]* 13.5 Write property tests for dynamic factor and Dixon-Coles (Properties 10-12)
    - **Property 10: Dynamic factor composite calculation**
    - **Property 11: Dixon-Coles score matrix validity**
    - **Property 12: Elo model probability completeness**
    - Create `tests/properties/test_prop_dynamic_factor.py` and `tests/properties/test_prop_dixon_coles.py`
    - Use Hypothesis strategies for streaks, rest days, Elo ratings, goal averages
    - **Validates: Requirements 7.1, 7.2, 7.5, 7.6, 7.7**

  - [x]* 13.6 Write property tests for coach style system (Properties 14-16)
    - **Property 14: Analyst style identity**
    - **Property 15: Contrarian style underdog boost with probability preservation**
    - **Property 16: Coach style narrative prefix**
    - Create `tests/properties/test_prop_coach_style.py`
    - Use Hypothesis strategies for predictions and coach styles
    - **Validates: Requirements 8.1, 8.2, 8.7**

  - [x]* 13.7 Write property tests for data integrity (Properties 17-18)
    - **Property 17: Atomic write data integrity**
    - **Property 18: Prediction log entry completeness**
    - Create `tests/properties/test_prop_data_integrity.py`
    - Use Hypothesis strategies for JSON data objects and prediction events
    - **Validates: Requirements 9.5, 9.6**

  - [x]* 13.8 Write property tests for accuracy and streaks (Properties 19-20)
    - **Property 19: Accuracy metric calculation correctness**
    - **Property 20: Streak counter update correctness**
    - Create `tests/properties/test_prop_accuracy.py` and `tests/properties/test_prop_streaks.py`
    - Use Hypothesis strategies for result lists and match sequences
    - **Validates: Requirements 4.5, 5.1, 5.2, 5.3**

- [x] 14. Write integration tests
  - [x]* 14.1 Write integration tests for MCP tool end-to-end flows
    - Test full predict_match flow (valid input → formatted output)
    - Test full predict_group flow (valid group → table output)
    - Test predict_champion flow (simulation → bracket output)
    - Test update_results flow (result → recalibration → report)
    - Test accuracy_stats flow (sufficient/insufficient data scenarios)
    - Test team_info flow (exact match, fuzzy match, no match)
    - Create `tests/integration/test_mcp_tools.py`
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1_

  - [x]* 14.2 Write integration tests for startup validation
    - Test successful startup with valid data files
    - Test startup abort on missing teams, invalid group counts, missing fields
    - Test fallback data behavior on external source timeout
    - Create `tests/integration/test_startup_validation.py`
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x]* 14.3 Write unit tests for edge cases and error handling
    - Test invalid team name returns ≤ 3 suggestions
    - Test invalid group ID returns all valid options
    - Test invalid coach style returns three valid options
    - Test sub-model failure triggers graceful degradation
    - Test accuracy report with 0 matches and < 3 matches
    - Test recalibration systematic bias report trigger (< 50% direction accuracy after 5 matches)
    - _Requirements: 1.5, 1.7, 2.4, 5.7, 5.8, 7.8, 4.8_

- [x] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The project uses Python with FastMCP SDK, NumPy for numerical computation, Rich for CLI output, and Hypothesis for property-based testing
- All 20 correctness properties from the design document are covered by property test tasks (section 13)
- Core implementation (tasks 1-11) is complete; remaining work is testing

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.4", "2.5", "4.5"] },
    { "id": 1, "tasks": ["5.3", "6.3"] },
    { "id": 2, "tasks": ["8.3"] },
    { "id": 3, "tasks": ["13.1", "13.2", "13.5"] },
    { "id": 4, "tasks": ["13.3", "13.4", "13.6"] },
    { "id": 5, "tasks": ["13.7", "13.8"] },
    { "id": 6, "tasks": ["14.1", "14.2", "14.3"] }
  ]
}
```
