"""
Constants for the FIFA Predictor Power.

Contains all 48 team names, aliases, confederation mappings,
host nations, group assignments, and model default parameters.
"""

# ============================================================================
# HOST NATIONS (2026 World Cup)
# ============================================================================

HOST_NATIONS: set[str] = {"United States", "Canada", "Mexico"}

# ============================================================================
# DEFAULT MODEL WEIGHTS
# ============================================================================

DEFAULT_WEIGHTS: dict[str, float] = {
    "poisson": 0.40,
    "elo": 0.25,
    "h2h": 0.15,
    "dynamic": 0.20,
}

# ============================================================================
# LEAGUE AVERAGE GOALS (World Cup historical average per team per match)
# ============================================================================

LEAGUE_AVG_GOALS: float = 1.35

# ============================================================================
# CONFEDERATION COEFFICIENTS
# Used in Dixon-Coles model to adjust attack strength by confederation quality
# ============================================================================

CONFEDERATION_COEFFICIENTS: dict[str, float] = {
    "UEFA": 1.05,
    "CONMEBOL": 1.03,
    "CONCACAF": 0.95,
    "CAF": 0.92,
    "AFC": 0.90,
    "OFC": 0.85,
}

# ============================================================================
# TEAM ALIASES
# Maps canonical English name -> list of aliases (Chinese, abbreviations, etc.)
# ============================================================================

TEAM_ALIASES: dict[str, list[str]] = {
    # UEFA (16 teams)
    "Germany": ["德國", "GER", "Deutschland"],
    "Spain": ["西班牙", "ESP", "España"],
    "France": ["法國", "FRA"],
    "Portugal": ["葡萄牙", "POR"],
    "England": ["英格蘭", "ENG"],
    "Netherlands": ["荷蘭", "NED", "Holland"],
    "Belgium": ["比利時", "BEL"],
    "Italy": ["義大利", "ITA", "Italia"],
    "Croatia": ["克羅埃西亞", "CRO", "Hrvatska"],
    "Switzerland": ["瑞士", "SUI"],
    "Austria": ["奧地利", "AUT"],
    "Denmark": ["丹麥", "DEN", "Danmark"],
    "Serbia": ["塞爾維亞", "SRB", "Srbija"],
    "Scotland": ["蘇格蘭", "SCO"],
    "Slovenia": ["斯洛維尼亞", "SVN"],
    "Poland": ["波蘭", "POL", "Polska"],
    # CONMEBOL (6+1 teams, includes intercontinental playoff winner)
    "Brazil": ["巴西", "BRA", "Brasil"],
    "Argentina": ["阿根廷", "ARG"],
    "Uruguay": ["烏拉圭", "URU"],
    "Colombia": ["哥倫比亞", "COL"],
    "Ecuador": ["厄瓜多", "ECU"],
    "Paraguay": ["巴拉圭", "PAR"],
    "Peru": ["秘魯", "PER", "Perú"],
    # CONCACAF (6 teams)
    "United States": ["美國", "USA", "US"],
    "Mexico": ["墨西哥", "MEX", "México"],
    "Canada": ["加拿大", "CAN"],
    "Costa Rica": ["哥斯大黎加", "CRC"],
    "Jamaica": ["牙買加", "JAM"],
    "Honduras": ["宏都拉斯", "HON"],
    # CAF (9 teams)
    "Morocco": ["摩洛哥", "MAR", "Maroc"],
    "Senegal": ["塞內加爾", "SEN"],
    "Nigeria": ["奈及利亞", "NGA"],
    "Cameroon": ["喀麥隆", "CMR", "Cameroun"],
    "Egypt": ["埃及", "EGY"],
    "South Africa": ["南非", "RSA"],
    "Algeria": ["阿爾及利亞", "ALG", "Algérie"],
    "Mali": ["馬利", "MLI"],
    "Tunisia": ["突尼西亞", "TUN", "Tunisie"],
    # AFC (8+1 teams)
    "Japan": ["日本", "JPN"],
    "South Korea": ["南韓", "KOR", "Korea Republic"],
    "Australia": ["澳洲", "AUS"],
    "Saudi Arabia": ["沙烏地阿拉伯", "KSA", "沙特"],
    "Iran": ["伊朗", "IRN", "IR Iran"],
    "Qatar": ["卡達", "QAT"],
    "Iraq": ["伊拉克", "IRQ"],
    "Uzbekistan": ["烏茲別克", "UZB"],
    "Indonesia": ["印尼", "IDN", "印度尼西亞"],
    # OFC (1 team)
    "New Zealand": ["紐西蘭", "NZL"],
}

# ============================================================================
# CONFEDERATION MAP
# Maps team canonical name -> confederation
# ============================================================================

CONFEDERATION_MAP: dict[str, str] = {
    # UEFA
    "Germany": "UEFA",
    "Spain": "UEFA",
    "France": "UEFA",
    "Portugal": "UEFA",
    "England": "UEFA",
    "Netherlands": "UEFA",
    "Belgium": "UEFA",
    "Italy": "UEFA",
    "Croatia": "UEFA",
    "Switzerland": "UEFA",
    "Austria": "UEFA",
    "Denmark": "UEFA",
    "Serbia": "UEFA",
    "Scotland": "UEFA",
    "Slovenia": "UEFA",
    "Poland": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL",
    "Argentina": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    # CONCACAF
    "United States": "CONCACAF",
    "Mexico": "CONCACAF",
    "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF",
    "Jamaica": "CONCACAF",
    "Honduras": "CONCACAF",
    # CAF
    "Morocco": "CAF",
    "Senegal": "CAF",
    "Nigeria": "CAF",
    "Cameroon": "CAF",
    "Egypt": "CAF",
    "South Africa": "CAF",
    "Algeria": "CAF",
    "Mali": "CAF",
    "Tunisia": "CAF",
    # AFC
    "Japan": "AFC",
    "South Korea": "AFC",
    "Australia": "AFC",
    "Saudi Arabia": "AFC",
    "Iran": "AFC",
    "Qatar": "AFC",
    "Iraq": "AFC",
    "Uzbekistan": "AFC",
    "Indonesia": "AFC",
    # OFC
    "New Zealand": "OFC",
}

# ============================================================================
# GROUP ASSIGNMENTS (A-L, 4 teams per group)
# Based on 2026 FIFA World Cup draw results
# ============================================================================

GROUP_ASSIGNMENTS: dict[str, list[str]] = {
    "A": ["United States", "Morocco", "Ecuador", "Jamaica"],
    "B": ["Spain", "Portugal", "Uruguay", "Paraguay"],
    "C": ["Brazil", "France", "Colombia", "Honduras"],
    "D": ["Argentina", "England", "Nigeria", "Costa Rica"],
    "E": ["Germany", "Italy", "South Korea", "New Zealand"],
    "F": ["Netherlands", "Belgium", "Senegal", "Qatar"],
    "G": ["Japan", "Croatia", "Serbia", "Cameroon"],
    "H": ["Mexico", "Canada", "Egypt", "Indonesia"],
    "I": ["Denmark", "Switzerland", "Iran", "Scotland"],
    "J": ["Austria", "Poland", "Algeria", "Mali"],
    "K": ["Australia", "Saudi Arabia", "Tunisia", "Slovenia"],
    "L": ["South Africa", "Iraq", "Uzbekistan", "Peru"],
}

# ============================================================================
# MODEL DEFAULT PARAMETERS
# ============================================================================

# Dixon-Coles rho parameter (low-score correction factor)
DIXON_COLES_RHO: float = -0.13

# Neutral venue factor (no home advantage)
NEUTRAL_VENUE_FACTOR: float = 1.0

# Host nation Elo bonus when playing in their country
HOST_NATION_ELO_BONUS: int = 50

# Dynamic factor thresholds and values
STREAK_THRESHOLD: int = 3
STREAK_BONUS: float = 0.05
FATIGUE_DAYS_THRESHOLD: int = 3
FATIGUE_PENALTY: float = -0.03
REVENGE_BONUS: float = 0.03

# Ensemble weight constraints
WEIGHT_MIN: float = 0.10
WEIGHT_MAX: float = 0.60
MAX_WEIGHT_ADJUSTMENT: float = 0.05

# Monte Carlo default simulations
DEFAULT_SIMULATIONS: int = 10000

# Accuracy tracker minimum sample size
MIN_ACCURACY_SAMPLE_SIZE: int = 3

# Over/Under threshold
OVER_UNDER_THRESHOLD: float = 2.5

# Score matrix size (0-4 goals each team)
SCORE_MATRIX_SIZE: int = 5

# Confidence index ranges
CONFIDENCE_LOW_MAX: int = 33
CONFIDENCE_MID_MAX: int = 66
CONFIDENCE_HIGH_MAX: int = 100

# ============================================================================
# ALL TEAM NAMES (convenience list)
# ============================================================================

ALL_TEAMS: list[str] = sorted(CONFEDERATION_MAP.keys())
