# morie.fn -- function file (hadesllm/morie)
"""Poisson regression with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
import statsmodels.api as sm


def glmpoi(X: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray],
           add_intercept: bool = True):
    """Poisson regression (GLM with log link)."""
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
    if np.any(y < 0):
        raise ValueError("Poisson y must be non-negative.")
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    coef_rows = []
    for i, (c, se, p) in enumerate(zip(model.params, model.bse, model.pvalues)):
        name = "(Intercept)" if i == 0 and add_intercept else f"x{i}"
        coef_rows.append([name, f"{float(c):.4g}", f"{float(se):.4g}",
                          f"{float(p):.4g}",
                          "***" if p < 0.001 else "**" if p < 0.01 else
                          "*" if p < 0.05 else "."  if p < 0.1 else ""])
    fitted = model.fittedvalues
    pearson_chi2 = float(np.sum((y - fitted) ** 2 / fitted)) if np.all(fitted > 0) else float("nan")
    overdispersion = pearson_chi2 / (len(y) - len(model.params)) if len(y) > len(model.params) else float("nan")
    warnings = []
    if not np.isnan(overdispersion) and overdispersion > 1.5:
        warnings.append(f"overdispersion: Pearson chi^2/df = {overdispersion:.2f} > 1.5; "
                        "Poisson assumes equidispersion. Consider negative binomial.")
    return RichResult(
        title="Poisson regression (GLM, log link)",
        summary_lines=[
            ("AIC", float(model.aic)), ("Deviance", float(model.deviance)),
            ("Pearson chi^2", pearson_chi2), ("Pearson chi^2 / df", overdispersion),
            ("n observations", len(y)), ("n parameters", len(model.params)),
            ("Log-likelihood", float(model.llf)),
        ],
        tables=[{
            "title": "Coefficients (link scale = log mean):",
            "headers": ["Variable", "Estimate", "Std. Error", "Pr(>|z|)", "Sig."],
            "rows": coef_rows,
        }],
        warnings=warnings,
        interpretation=("Significance codes: 0 *** 0.001 ** 0.01 * 0.05 . 0.1. "
                        "Coefficients are on log-mean scale; exp(coef) = rate ratio."),
        payload={"coef": model.params.tolist(), "pvalues": model.pvalues.tolist(),
                 "aic": float(model.aic), "deviance": float(model.deviance),
                 "fitted": fitted.tolist(), "overdispersion": overdispersion},
    )
