"""
Unit tests for InputValidator.

Tests validate_team(), validate_group(), and validate_coach_style() methods.
"""

import pytest

from src.data.data_manager import PredictionError
from src.utils.validator import CoachStyleType, InputValidator


@pytest.fixture
def validator() -> InputValidator:
    """Create an InputValidator with default TeamMatcher."""
    return InputValidator()


# ============================================================================
# validate_team tests
# ============================================================================


class TestValidateTeam:
    """Tests for InputValidator.validate_team()."""

    def test_exact_english_name(self, validator: InputValidator) -> None:
        """Exact English canonical name should return the name."""
        result = validator.validate_team("Brazil")
        assert result == "Brazil"

    def test_exact_english_name_case_insensitive(self, validator: InputValidator) -> None:
        """Case-insensitive match on English name."""
        result = validator.validate_team("brazil")
        assert result == "Brazil"

    def test_chinese_name(self, validator: InputValidator) -> None:
        """Chinese name should resolve to canonical name."""
        result = validator.validate_team("巴西")
        assert result == "Brazil"

    def test_abbreviation(self, validator: InputValidator) -> None:
        """Abbreviation should resolve to canonical name."""
        result = validator.validate_team("BRA")
        assert result == "Brazil"

    def test_alternate_name(self, validator: InputValidator) -> None:
        """Alternate names should resolve correctly."""
        result = validator.validate_team("USA")
        assert result == "United States"

    def test_invalid_team_returns_error(self, validator: InputValidator) -> None:
        """Invalid team name should return PredictionError."""
        result = validator.validate_team("Atlantis")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        assert len(result.suggestions) <= 3

    def test_empty_string_returns_error(self, validator: InputValidator) -> None:
        """Empty string should return error."""
        result = validator.validate_team("")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"

    def test_whitespace_only_returns_error(self, validator: InputValidator) -> None:
        """Whitespace-only input should return error."""
        result = validator.validate_team("   ")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"

    def test_similar_name_provides_suggestions(self, validator: InputValidator) -> None:
        """A not-quite-matching name should provide suggestions."""
        result = validator.validate_team("Brzzl")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_TEAM"
        # Should suggest similar team names
        assert len(result.suggestions) > 0
        assert len(result.suggestions) <= 3


# ============================================================================
# validate_group tests
# ============================================================================


class TestValidateGroup:
    """Tests for InputValidator.validate_group()."""

    def test_uppercase_group(self, validator: InputValidator) -> None:
        """Uppercase group ID should be valid."""
        assert validator.validate_group("A") == "A"
        assert validator.validate_group("L") == "L"

    def test_lowercase_group(self, validator: InputValidator) -> None:
        """Lowercase group ID should be normalized to uppercase."""
        assert validator.validate_group("a") == "A"
        assert validator.validate_group("l") == "L"

    def test_all_valid_groups(self, validator: InputValidator) -> None:
        """All groups A-L should be valid."""
        for group_char in "ABCDEFGHIJKL":
            result = validator.validate_group(group_char)
            assert result == group_char

    def test_invalid_group_returns_error(self, validator: InputValidator) -> None:
        """Invalid group returns PredictionError with all valid groups."""
        result = validator.validate_group("M")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"
        assert sorted(result.suggestions) == list("ABCDEFGHIJKL")

    def test_empty_group_returns_error(self, validator: InputValidator) -> None:
        """Empty string should return error."""
        result = validator.validate_group("")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"

    def test_numeric_group_returns_error(self, validator: InputValidator) -> None:
        """Numeric input should return error."""
        result = validator.validate_group("1")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_GROUP"

    def test_whitespace_trimmed(self, validator: InputValidator) -> None:
        """Whitespace around group ID should be trimmed."""
        assert validator.validate_group("  B  ") == "B"


# ============================================================================
# validate_coach_style tests
# ============================================================================


class TestValidateCoachStyle:
    """Tests for InputValidator.validate_coach_style()."""

    # Direct Chinese names
    def test_chinese_analyst(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("分析師") == CoachStyleType.ANALYST

    def test_chinese_contrarian(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("反向思考者") == CoachStyleType.CONTRARIAN

    def test_chinese_tactician(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("戰術家") == CoachStyleType.TACTICIAN

    # Direct English names
    def test_english_analyst(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("analyst") == CoachStyleType.ANALYST

    def test_english_contrarian(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("contrarian") == CoachStyleType.CONTRARIAN

    def test_english_tactician(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("tactician") == CoachStyleType.TACTICIAN

    # Case-insensitive English names
    def test_english_case_insensitive(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("ANALYST") == CoachStyleType.ANALYST
        assert validator.validate_coach_style("Contrarian") == CoachStyleType.CONTRARIAN
        assert validator.validate_coach_style("TACTICIAN") == CoachStyleType.TACTICIAN

    # English keywords
    def test_keyword_conservative(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("conservative") == CoachStyleType.ANALYST

    def test_keyword_aggressive(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("aggressive") == CoachStyleType.CONTRARIAN

    def test_keyword_balanced(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("balanced") == CoachStyleType.TACTICIAN

    # Chinese keywords
    def test_keyword_chinese_conservative(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("保守") == CoachStyleType.ANALYST

    def test_keyword_chinese_aggressive(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("激進") == CoachStyleType.CONTRARIAN

    def test_keyword_chinese_balanced(self, validator: InputValidator) -> None:
        assert validator.validate_coach_style("平衡") == CoachStyleType.TACTICIAN

    # Invalid styles
    def test_invalid_style_returns_error(self, validator: InputValidator) -> None:
        result = validator.validate_coach_style("random")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_STYLE"
        assert len(result.suggestions) == 3

    def test_empty_style_returns_error(self, validator: InputValidator) -> None:
        result = validator.validate_coach_style("")
        assert isinstance(result, PredictionError)
        assert result.error_code == "INVALID_STYLE"

    def test_whitespace_trimmed(self, validator: InputValidator) -> None:
        """Whitespace around style should be trimmed."""
        assert validator.validate_coach_style("  analyst  ") == CoachStyleType.ANALYST
