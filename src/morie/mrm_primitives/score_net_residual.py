"""Two-stage score-then-audit primitive.

Adapted from O'Connell & Laniyonu (2025) Race & Justice 15(3):428–453.
Many institutional decisions are SUPPOSED to be made on the basis of
an actuarial score (a risk-assessment instrument, a credit score, an
education-readiness flag, etc.).  Policy claims that the score is
the binding input — so any residual race / gender effect on the
downstream outcome, NET of the score, is by-construction evidence of
disparate treatment / staff over-ride bias.

The primitive runs the two-stage regression:

    1. score ~ race + gender + age + ...     (stage 1, optional —
                                              already in the data)
    2. outcome ~ score + race + gender + ... (stage 2 — the audit)

and returns the stage-2 race/gender coefficients with their standard
errors.  The residual race coefficient is the "disparate treatment"
signal: a non-zero value indicates the institution is NOT actually
treating people of equal scores equally.

Caveat (per O'Connell & Laniyonu): this is *output* disparity, not
*predictive-validity* disparity.  A non-zero residual race effect
could also be evidence of valid race-difference in the unobserved
outcome — Goel et al. (2021) frame this distinction formally and
warn against equating output bias with predictive bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScoreNetResidualResult:
    """Stage-2 audit coefficients."""
    sensitive_attribute_names: list[str]
    coefficients: dict[str, float]               # name -> stage-2 coefficient
    std_errors: dict[str, float]                 # name -> bootstrap SE
    n_obs: int
    score_coefficient: float
    interpretation: str

    def is_significant(self, attribute: str, alpha: float = 0.05) -> bool:
        """Two-sided Wald test approximation."""
        b = self.coefficients.get(attribute)
        se = self.std_errors.get(attribute)
        if b is None or se is None or se == 0:
            return False
        z = abs(b / se)
        # Quick approx: |z| >= 1.96 for alpha=0.05; 2.58 for alpha=0.01
        critical = 1.959964 if alpha == 0.05 else (2.575829 if alpha == 0.01 else
                                                    abs(_norm_ppf(1 - alpha / 2)))
        return z >= critical


def score_net_residual(
    df: pd.DataFrame,
    *,
    score_col: str,
    outcome_col: str,
    sensitive_cols: list[str],
    control_cols: list[str] | None = None,
    family: str = "logit",
    bootstrap_replicates: int = 200,
    random_state: int = 20260513,
) -> ScoreNetResidualResult:
    """Two-stage score-then-audit residual race effect.

    Parameters
    ----------
    df : pd.DataFrame
        Sample microdata.
    score_col : str
        The actuarial score the institution claims to use.
    outcome_col : str
        Binary outcome (parole granted, loan approved, etc.) — 0/1.
    sensitive_cols : list[str]
        Race / gender / etc. indicators to test for residual bias.
        Pre-dummy categorical attributes.
    control_cols : list[str], optional
        Additional controls (age, prior offences, etc.).
    family : {"logit", "linear"}, default "logit"
        Stage-2 model.  Use ``"logit"`` for binary outcomes,
        ``"linear"`` for continuous.
    bootstrap_replicates : int, default 200
        Resamples for SEs.  Set to 0 to skip (returns SE = nan).
    random_state : int, default 20260513
        RNG seed for reproducibility.

    Returns
    -------
    ScoreNetResidualResult
    """
    if control_cols is None:
        control_cols = []

    # Stack predictors: score + sensitive + controls
    feature_cols = [score_col, *sensitive_cols, *control_cols]
    Xy = df[feature_cols + [outcome_col]].dropna()
    X = Xy[feature_cols].to_numpy(dtype=float)
    y = Xy[outcome_col].to_numpy(dtype=float)
    n = X.shape[0]

    coef = _fit(X, y, family)
    # coef[0] = intercept; coef[1] = score; coef[2..] = sensitive then controls
    score_coef = float(coef[2])  # idx 1 is intercept after column-stacking 1s; idx 2 is score
    # Actually: _fit prepends a 1 column; so coef[0]=intercept, coef[1]=score,
    # coef[2..1+len(sensitive)] = sensitive coefs.
    sensitive_idx = list(range(2, 2 + len(sensitive_cols)))
    coefs_dict = {name: float(coef[i])
                  for name, i in zip(sensitive_cols, sensitive_idx)}

    if bootstrap_replicates > 0:
        rng = np.random.default_rng(random_state)
        boot_coefs = np.zeros((bootstrap_replicates, X.shape[1] + 1))
        for b in range(bootstrap_replicates):
            idx = rng.integers(0, n, size=n)
            try:
                boot_coefs[b] = _fit(X[idx], y[idx], family)
            except (ValueError, np.linalg.LinAlgError):
                boot_coefs[b] = np.nan
        ses = np.nanstd(boot_coefs, axis=0)
        se_dict = {name: float(ses[i])
                   for name, i in zip(sensitive_cols, sensitive_idx)}
    else:
        se_dict = {name: float("nan") for name in sensitive_cols}

    # Build a one-line interpretation
    bias_lines = []
    for name in sensitive_cols:
        b = coefs_dict[name]
        se = se_dict[name]
        marker = "*" if (not np.isnan(se) and abs(b) >= 1.96 * se) else ""
        bias_lines.append(f"{name}={b:+.4f}({se:.4f}){marker}")
    interp = (
        f"Residual sensitive-attribute coefficients net of score "
        f"({score_col}, β={score_coef:+.4f}): " + ", ".join(bias_lines)
        + ".  * = |β|/SE >= 1.96.  Significant residual = institution "
        "is not using the score uniformly across groups."
    )

    return ScoreNetResidualResult(
        sensitive_attribute_names=sensitive_cols,
        coefficients=coefs_dict,
        std_errors=se_dict,
        n_obs=n,
        score_coefficient=score_coef,
        interpretation=interp,
    )


# ─── helpers ────────────────────────────────────────────────────────────


def _fit(X: np.ndarray, y: np.ndarray, family: str) -> np.ndarray:
    """Single-shot regression fit; returns coef with intercept first."""
    if family == "linear":
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        return np.linalg.lstsq(X_int, y, rcond=None)[0]
    if family == "logit":
        from .synthetic_exposure import _logistic_fit
        return _logistic_fit(X, y)
    raise ValueError(f"unknown family {family!r}; expected 'logit' or 'linear'")


def _norm_ppf(p: float) -> float:
    """Beasley-Springer approximation to the inverse normal CDF.
    Used only when caller passes alpha != 0.05 or 0.01.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if 0.0 < p < p_low:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = (-2 * np.log(1 - p)) ** 0.5
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
