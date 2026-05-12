"""OTIS domain constants for correctional placement analysis."""

from __future__ import annotations

REGIONS: list[str] = ["Central", "Eastern", "Northern", "Toronto", "Western"]

AGE_GROUPS: list[str] = ["18 to 24", "25 to 49", "50+"]

AGE_NUMERIC: dict[str, float] = {
    "18 to 24": 21.0,
    "25 to 49": 42.0,
    "50+": 57.5,
}

GENDERS: list[str] = ["Male", "Female", "Non-binary"]

# 8 possible alert-state combinations (3 binary alerts)
# binA=mental_health, binB=suicide_risk, binC=suicide_watch
ALERT_COMBOS: dict[str, tuple[int, int, int]] = {
    "a1": (1, 0, 0),  # mental health only
    "a2": (0, 1, 0),  # suicide risk only
    "a3": (0, 0, 1),  # suicide watch only
    "a4": (1, 1, 0),  # mental health + suicide risk
    "a5": (0, 1, 1),  # suicide risk + suicide watch
    "a6": (1, 0, 1),  # mental health + suicide watch
    "a7": (1, 1, 1),  # all three alerts
    "a8": (0, 0, 0),  # no alerts
}

ALERT_NAMES: dict[str, str] = {
    "a1": "Mental health only",
    "a2": "Suicide risk only",
    "a3": "Suicide watch only",
    "a4": "Mental health + suicide risk",
    "a5": "Suicide risk + suicide watch",
    "a6": "Mental health + suicide watch",
    "a7": "All three alerts",
    "a8": "No alerts",
}

FACILITY_TYPES: list[str] = [
    "Provincial Correctional Centre",
    "Provincial Detention Centre",
    "Community Supervision",
    "Halfway House",
    "Treatment Centre",
]

SECURITY_LEVELS: list[str] = ["Minimum", "Medium", "Maximum", "Protective Custody"]

OFFENSE_CATEGORIES: list[str] = [
    "Violent",
    "Property",
    "Drug",
    "Sexual",
    "Administration of Justice",
    "Other Criminal Code",
    "Provincial Statute",
]

RISK_LEVELS: list[str] = ["Low", "Medium", "High", "Very High"]

RISK_TOOLS: list[str] = ["LSI-R", "Static-99", "COMPAS", "Ontario Domestic Assault Risk Assessment"]

PROGRAM_TYPES: list[str] = [
    "Anger Management",
    "Substance Abuse",
    "Cognitive Behavioural",
    "Life Skills",
    "Educational/Vocational",
    "Mental Health",
    "Sex Offender Treatment",
    "Domestic Violence",
    "Reintegration",
]

RELEASE_TYPES: list[str] = [
    "Parole",
    "Statutory Release",
    "Warrant Expiry",
    "Temporary Absence",
    "Bail",
    "Court Order",
]

# Default column names (dataset-agnostic -- override via keyword args)
DEFAULT_COLS: dict[str, str] = {
    "id": "unique_individual_id",
    "year": "end_fiscal_year",
    "region": "region",
    "age": "age_group",
    "gender": "gender",
    "alert_mh": "alert_mental_health",
    "alert_sr": "alert_suicide_risk",
    "alert_sw": "alert_suicide_watch",
    "facility": "facility_type",
    "security": "security_level",
    "offense": "offense_category",
    "risk": "risk_level",
    "sentence": "sentence_days",
    "outcome": "Y",
    "treatment": "D",
}
