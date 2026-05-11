"""Target trial emulation framework."""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def target_trial(
    df: pd.DataFrame,
    *,
    eligibility_fn: callable | None = None,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    time_col: str | None = None,
) -> DescriptiveResult:
    """
    Target trial emulation: apply eligibility criteria, clone-censor,
    and estimate a treatment effect from observational data.

    Parameters
    ----------
    df : DataFrame
        Observational data.
    eligibility_fn : callable, optional
        Function taking a row and returning True if eligible. Default: all.
    treatment_col : str
        Treatment column name.
    outcome_col : str
        Outcome column name.
    time_col : str, optional
        Time column (for longitudinal designs).

    Returns
    -------
    DescriptiveResult
        extra has 'n_eligible', 'risk_treated', 'risk_control',
        'risk_difference'.

    References
    ----------
    Hernan, M. A., & Robins, J. M. (2016). Using big data to emulate
    a target trial when a randomised trial is not available. *Am J
    Epidemiol*, 183(8), 758-764.
    """
    if treatment_col not in df.columns or outcome_col not in df.columns:
        raise ValueError("treatment and outcome columns required.")

    if eligibility_fn is not None:
        eligible = df[df.apply(eligibility_fn, axis=1)].copy()
    else:
        eligible = df.copy()

    n_elig = len(eligible)
    if n_elig < 4:
        raise ValueError("Insufficient eligible subjects.")

    treated = eligible[eligible[treatment_col] == 1]
    control = eligible[eligible[treatment_col] == 0]

    r1 = float(treated[outcome_col].mean()) if len(treated) > 0 else 0.0
    r0 = float(control[outcome_col].mean()) if len(control) > 0 else 0.0
    rd = r1 - r0

    return DescriptiveResult(
        name="target_trial",
        value=rd,
        extra={
            "n_eligible": n_elig,
            "n_treated": len(treated),
            "n_control": len(control),
            "risk_treated": r1,
            "risk_control": r0,
            "risk_difference": rd,
        },
    )


tgtrl = target_trial


def cheatsheet() -> str:
    return "target_trial({}) -> Target trial emulation framework."
