"""
Input validation utility for the FIFA Predictor Power.

Validates team names, group IDs, and coach styles,
returning structured PredictionError on failure with helpful suggestions.
"""

from enum import Enum
from typing import Optional, TYPE_CHECKING

from src.data.data_manager import PredictionError
from src.utils.constants import ALL_TEAMS, ALL_KNOWN_TEAMS, GROUP_ASSIGNMENTS, NON_PARTICIPANT_ALIASES

if TYPE_CHECKING:
    from src.utils.team_matcher import TeamMatcher


# ============================================================================
# Coach Style Type (defined here for self-containment until coach_style.py exists)
# ============================================================================


class CoachStyleType(Enum):
    """Three coaching analysis perspectives."""

    ANALYST = "分析師"
    CONTRARIAN = "反向思考者"
    TACTICIAN = "戰術家"


# Direct name mappings (Chinese and English, case-insensitive)
_DIRECT_STYLE_NAMES: dict[str, CoachStyleType] = {
    "分析師": CoachStyleType.ANALYST,
    "反向思考者": CoachStyleType.CONTRARIAN,
    "戰術家": CoachStyleType.TACTICIAN,
    "analyst": CoachStyleType.ANALYST,
    "contrarian": CoachStyleType.CONTRARIAN,
    "tactician": CoachStyleType.TACTICIAN,
}

# Keyword mappings (Chinese and English, case-insensitive)
_STYLE_KEYWORDS: dict[str, CoachStyleType] = {
    "conservative": CoachStyleType.ANALYST,
    "保守": CoachStyleType.ANALYST,
    "aggressive": CoachStyleType.CONTRARIAN,
    "激進": CoachStyleType.CONTRARIAN,
    "balanced": CoachStyleType.TACTICIAN,
    "平衡": CoachStyleType.TACTICIAN,
}

# Valid group IDs
_VALID_GROUPS: set[str] = set("ABCDEFGHIJKL")


class InputValidator:
    """
    Validates user inputs for the FIFA Predictor Power.

    Uses TeamMatcher for team name resolution (dependency injection).
    Falls back to exact matching against ALL_TEAMS if no TeamMatcher is provided.
    Supports both World Cup participants and non-participant teams (for friendlies).
    """

    def __init__(self, team_matcher: Optional["TeamMatcher"] = None):
        """
        Initialize the validator.

        Args:
            team_matcher: Optional TeamMatcher instance for fuzzy team name matching.
                          If not provided, a default one will be created on first use.
        """
        self._team_matcher = team_matcher

    @property
    def team_matcher(self) -> "TeamMatcher":
        """Lazy-load TeamMatcher if not injected."""
        if self._team_matcher is None:
            from src.utils.team_matcher import TeamMatcher

            self._team_matcher = TeamMatcher(ALL_KNOWN_TEAMS)
        return self._team_matcher

    def validate_team(self, name: str) -> str | PredictionError:
        """
        Validate a team name input.

        Supports English names, Chinese names, abbreviations, and fuzzy matching.
        Recognizes both World Cup participants and non-participant teams.

        Args:
            name: The team name to validate.

        Returns:
            str: The normalized canonical team name on success.
            PredictionError: Error with up to 3 suggestions on failure.
        """
        if not name or not name.strip():
            return PredictionError(
                error_code="INVALID_TEAM",
                message="球隊名稱不得為空。",
                suggestions=[]
            )

        matcher = self.team_matcher
        result = matcher.match(name)

        if result.match_type == "exact":
            return result.team_name  # type: ignore[return-value]

        if result.match_type == "single":
            return result.team_name  # type: ignore[return-value]

        if result.match_type == "multiple":
            # Multiple candidates found — list all for user to choose (Req 6.4)
            candidates = result.candidates
            return PredictionError(
                error_code="INVALID_TEAM",
                message=f"找不到與「{name.strip()}」完全匹配的球隊，有多個相似結果。",
                suggestions=candidates
            )

        # No match
        suggestions = result.suggestions[:3] if result.suggestions else []
        return PredictionError(
            error_code="INVALID_TEAM",
            message=f"找不到球隊「{name.strip()}」。請確認球隊名稱是否正確。",
            suggestions=suggestions
        )

    def is_non_participant(self, team_name: str) -> bool:
        """Check if a team is a non-World Cup participant.

        Args:
            team_name: Canonical team name.

        Returns:
            True if the team is not in the 48 World Cup participants.
        """
        return team_name not in ALL_TEAMS

    def validate_group(self, group_id: str) -> str | PredictionError:
        """
        Validate a group ID (A-L, case-insensitive).

        Args:
            group_id: The group identifier to validate.

        Returns:
            str: The normalized uppercase group ID on success.
            PredictionError: Error listing all valid group IDs on failure.
        """
        if not group_id or not group_id.strip():
            return PredictionError(
                error_code="INVALID_GROUP",
                message="小組代號不得為空。",
                suggestions=sorted(_VALID_GROUPS)
            )

        normalized = group_id.strip().upper()

        if normalized in _VALID_GROUPS:
            return normalized

        return PredictionError(
            error_code="INVALID_GROUP",
            message=f"小組代號「{group_id.strip()}」無效，有效代號為 A 至 L。",
            suggestions=sorted(_VALID_GROUPS)
        )

    def validate_coach_style(self, style: str) -> CoachStyleType | PredictionError:
        """
        Validate a coach style input.

        Supports:
        - Direct names: "分析師", "反向思考者", "戰術家", "analyst", "contrarian", "tactician"
        - Keywords: "conservative"/"保守" → analyst, "aggressive"/"激進" → contrarian,
                    "balanced"/"平衡" → tactician

        Args:
            style: The coach style string to validate.

        Returns:
            CoachStyleType: The resolved coach style enum on success.
            PredictionError: Error listing three valid options on failure.
        """
        if not style or not style.strip():
            return PredictionError(
                error_code="INVALID_STYLE",
                message="教練風格不得為空。",
                suggestions=["分析師 (analyst)", "反向思考者 (contrarian)", "戰術家 (tactician)"]
            )

        style_lower = style.strip().lower()

        # Try direct name match (case-insensitive)
        for name, coach_type in _DIRECT_STYLE_NAMES.items():
            if style_lower == name.lower():
                return coach_type

        # Try keyword match (case-insensitive)
        for keyword, coach_type in _STYLE_KEYWORDS.items():
            if style_lower == keyword.lower():
                return coach_type

        return PredictionError(
            error_code="INVALID_STYLE",
            message=f"教練風格「{style.strip()}」無效。",
            suggestions=["分析師 (analyst)", "反向思考者 (contrarian)", "戰術家 (tactician)"]
        )
