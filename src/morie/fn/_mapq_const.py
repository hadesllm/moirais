"""MAPQ domain constants for psychometric analysis.

MAPQII: Modified Attitudes on Psychedelics Questionnaire (20 items, 4 subscales).
"""

from __future__ import annotations

# Subscale definitions
SUBSCALES: dict[str, list[str]] = {
    "EE": ["EE1", "EE2", "EE3", "EE4", "EE5"],  # Experiential Engagement
    "EA": ["EA1", "EA2", "EA3", "EA4", "EA5"],  # Epistemic Attitudes
    "UA": ["UA1", "UA2", "UA3", "UA4", "UA5"],  # Utilitarian Attitudes
    "ER": ["ER1", "ER2", "ER3", "ER4", "ER5"],  # Ethical Reservations
}

ALL_ITEMS: list[str] = [
    "EE1",
    "EE2",
    "EE3",
    "EE4",
    "EE5",
    "EA1",
    "EA2",
    "EA3",
    "EA4",
    "EA5",
    "UA1",
    "UA2",
    "UA3",
    "UA4",
    "UA5",
    "ER1",
    "ER2",
    "ER3",
    "ER4",
    "ER5",
]

N_ITEMS: int = 20
N_SUBSCALES: int = 4
ITEMS_PER_SUBSCALE: int = 5

# Likert scale (typical 5-point)
RESPONSE_OPTIONS: list[int] = [1, 2, 3, 4, 5]
MIN_SCORE: int = 1
MAX_SCORE: int = 5

# Subscale score ranges
SUBSCALE_MIN: int = 5  # 5 items * 1
SUBSCALE_MAX: int = 25  # 5 items * 5
TOTAL_MIN: int = 20  # 20 items * 1
TOTAL_MAX: int = 100  # 20 items * 5

# Reverse-scored items (if any -- update as needed)
REVERSE_ITEMS: list[str] = []

# Factor structure for CFA (4-factor model)
FACTOR_STRUCTURE: dict[str, list[str]] = SUBSCALES.copy()

# DIF grouping variables
DIF_GROUPS: list[str] = ["gender", "age_group", "education", "language"]

# IRT model defaults
IRT_MODELS: list[str] = ["1PL", "2PL", "3PL", "GRM", "PCM", "RSM", "NRM"]

# CFA fit index thresholds (Hu & Bentler, 1999)
FIT_THRESHOLDS: dict[str, float] = {
    "cfi_good": 0.95,
    "cfi_acceptable": 0.90,
    "tli_good": 0.95,
    "tli_acceptable": 0.90,
    "rmsea_good": 0.06,
    "rmsea_acceptable": 0.08,
    "srmr_good": 0.08,
}

# Measurement invariance delta-fit criteria (Chen, 2007)
MI_DELTA: dict[str, float] = {
    "delta_cfi": 0.01,  # |ΔCFI| ≤ 0.01
    "delta_rmsea": 0.015,  # |ΔRMSEA| ≤ 0.015
}

# DIF classification (ETS, 2009)
DIF_CLASSIFICATION: dict[str, tuple[float, float]] = {
    "A": (0.0, 1.0),  # negligible
    "B": (1.0, 1.5),  # moderate
    "C": (1.5, float("inf")),  # large
}

# Default column names (dataset-agnostic)
DEFAULT_COLS: dict[str, str] = {
    "gender": "gender",
    "age": "age_group",
    "education": "education",
    "language": "language",
    "total_score": "total_score",
}
