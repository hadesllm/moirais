# morie.fn — function file (hadesllm/morie)
"""
Group Average Treatment Effect (GATE) via AIPW within strata.

Implements ``estimate_gate`` — partitions data by a grouping variable and
estimates the ATE within each group using the AIPW doubly-robust estimator.
"""

from __future__ import annotations

import logging as _logging

import pandas as pd

from morie.fn.aipw import estimate_aipw

_logger = _logging.getLogger(__name__)


def estimate_gate(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    group_col: str,
    propensity_col: str | None = None,
) -> pd.DataFrame:
    r"""Estimate Group Average Treatment Effects (GATE) via AIPW within strata.

    Partitions the data by *group_col* and estimates the ATE within each
    group using the AIPW doubly-robust estimator.

    .. math::

        \\text{GATE}_g = \\mathbb{E}[Y(1) - Y(0) \\mid G = g]

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param group_col: Column defining groups/strata.
    :type group_col: str
    :param propensity_col: Pre-computed propensity score column (optional).
    :type propensity_col: str or None
    :return: DataFrame with columns: ``group``, ``ate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``n``.
    :rtype: pandas.DataFrame

    References
    ----------
    Imai, K., & Ratkovic, M. (2013). Estimating treatment effect
    heterogeneity in randomized program evaluation.
    *Annals of Applied Statistics*, 7(1), 443--470.

    Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2020).
    Generic machine learning inference on heterogeneous treatment effects in
    randomized experiments. *NBER Working Paper* 24678.
    """
    required_cols = [treatment, outcome, group_col, *covariates]
    frame = data[required_cols].dropna().copy()

    results: list[dict] = []
    for group_val, group_df in frame.groupby(group_col):
        if group_df[treatment].nunique() < 2:
            _logger.warning("GATE: skipping group '%s' -- no variation in treatment", group_val)
            continue

        try:
            aipw_result = estimate_aipw(
                group_df,
                treatment=treatment,
                outcome=outcome,
                covariates=covariates,
                outcome_model="linear",
            )
            results.append(
                {
                    "group": group_val,
                    "ate": aipw_result["ate"],
                    "se": aipw_result["se"],
                    "ci_lower": aipw_result["ci_lower"],
                    "ci_upper": aipw_result["ci_upper"],
                    "n": aipw_result["n"],
                }
            )
        except Exception as exc:
            _logger.warning("GATE: failed for group '%s': %s", group_val, exc)
            results.append(
                {
                    "group": group_val,
                    "ate": float("nan"),
                    "se": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "n": len(group_df),
                }
            )

    return pd.DataFrame(results)


gate = estimate_gate


def cheatsheet() -> str:
    return "estimate_gate({}) -> Group Average Treatment Effect (GATE) via AIPW within strata"
