# moirais.fn — function file (hadesllm/moirais)
"""Charlson comorbidity index."""

from ._containers import ESRes

_CHARLSON_WEIGHTS = {
    "mi": 1,
    "chf": 1,
    "pvd": 1,
    "cvd": 1,
    "dementia": 1,
    "copd": 1,
    "rheumatic": 1,
    "peptic_ulcer": 1,
    "mild_liver": 1,
    "diabetes_uncomplicated": 1,
    "diabetes_complicated": 2,
    "hemiplegia": 2,
    "renal": 2,
    "malignancy": 2,
    "moderate_liver": 3,
    "metastatic": 6,
    "aids": 6,
}


def charlson_comorbidity(
    condition_flags: dict[str, int],
    age: int | None = None,
) -> ESRes:
    """Compute Charlson Comorbidity Index (CCI).

    Parameters
    ----------
    condition_flags : dict
        Keys = condition names, values = 0/1.
    age : int or None
        If provided, adds age-based score (50-59: +1, 60-69: +2, etc.).

    Returns
    -------
    ESRes

    References
    ----------
    Charlson, M. E. et al. (1987). A new method of classifying
    prognostic comorbidity. Journal of Chronic Diseases, 40(5), 373-383.
    """
    score = 0
    for cond, flag in condition_flags.items():
        if flag and cond in _CHARLSON_WEIGHTS:
            score += _CHARLSON_WEIGHTS[cond]

    age_score = 0
    if age is not None:
        if age >= 80:
            age_score = 4
        elif age >= 70:
            age_score = 3
        elif age >= 60:
            age_score = 2
        elif age >= 50:
            age_score = 1

    total = score + age_score
    ten_yr_survival = max(0.0, 0.983 ** (2.71828 ** (total * 0.9)))

    return ESRes(
        measure="CCI",
        estimate=float(total),
        extra={"comorbidity_score": score, "age_score": age_score, "est_10yr_survival": float(ten_yr_survival)},
    )


cdccm = charlson_comorbidity


def cheatsheet() -> str:
    return "charlson_comorbidity({}) -> Charlson comorbidity index."
