# morie.fn -- function file (hadesllm/morie)
"""
Local Average Treatment Effect (LATE) via instrumental variables.

Implements ``estimate_late`` -- estimates the LATE using 2SLS (linearmodels
or statsmodels) with automatic fallback to the Wald estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_late(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    r"""Estimate the Local Average Treatment Effect (LATE) via instrumental variables.

    For a binary instrument :math:`Z`, the **Wald estimator** (simple IV) is:

    .. math::

        \\widehat{\\text{LATE}} = \\frac{\\text{Cov}(Y, Z)}{\\text{Cov}(T, Z)}
        = \\frac{\\bar{Y}_{Z=1} - \\bar{Y}_{Z=0}}{\\bar{T}_{Z=1} - \\bar{T}_{Z=0}}

    With covariates, the function attempts 2SLS via ``linearmodels.iv.IV2SLS``
    or ``statsmodels`` IV regression, falling back to the Wald estimator if
    neither IV library is installed.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Endogenous treatment column.
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param instrument: Instrument column.
    :type instrument: str
    :param covariates: Exogenous covariate column names (optional).
    :type covariates: list[str] or None
    :return: Dictionary with ``late``, ``se``, ``ci`` (tuple), ``f_stat``,
        ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation
    of local average treatment effects. *Econometrica*, 62(2), 467--475.
    https://doi.org/10.2307/2951620

    Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of
    causal effects using instrumental variables. *JASA*, 91(434), 444--455.
    """
    cols = [treatment, outcome, instrument]
    if covariates:
        cols.extend(covariates)
    frame = data[cols].dropna().copy()

    y = frame[outcome].values.astype(float)
    t = frame[treatment].values.astype(float)
    z = frame[instrument].values.astype(float)
    n = len(frame)

    # Try linearmodels IV2SLS first
    try:
        from linearmodels.iv import IV2SLS as LM_IV2SLS

        if covariates:
            exog = sm.add_constant(frame[covariates].values.astype(float))
        else:
            exog = np.ones((n, 1))

        result = LM_IV2SLS(
            dependent=y,
            exog=exog,
            endog=t.reshape(-1, 1),
            instruments=z.reshape(-1, 1),
        ).fit(cov_type="robust")

        late_val = float(result.params.iloc[-1] if hasattr(result.params, "iloc") else result.params[-1])
        se_val = float(result.std_errors.iloc[-1] if hasattr(result.std_errors, "iloc") else result.std_errors[-1])
        try:
            f_stat = float(result.first_stage.diagnostics.iloc[0]["f.stat"])
        except Exception:
            f_stat = float("nan")

        z_crit = 1.959964
        return {
            "late": late_val,
            "se": se_val,
            "ci": (late_val - z_crit * se_val, late_val + z_crit * se_val),
            "f_stat": f_stat,
            "n": n,
            "method": "2SLS (linearmodels)",
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Try statsmodels IV2SLS
    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS as SM_IV2SLS

        if covariates:
            exog = np.column_stack(
                [
                    np.ones(n),
                    frame[covariates].values.astype(float),
                    t,
                ]
            )
            instrument_matrix = np.column_stack(
                [
                    np.ones(n),
                    frame[covariates].values.astype(float),
                    z,
                ]
            )
        else:
            exog = sm.add_constant(t)
            instrument_matrix = sm.add_constant(z)

        result = SM_IV2SLS(y, exog, instrument_matrix).fit()
        late_val = float(result.params[-1])
        se_val = float(result.bse[-1])

        # First-stage F-stat
        if covariates:
            X_first = np.column_stack([np.ones(n), frame[covariates].values.astype(float), z])
        else:
            X_first = sm.add_constant(z)
        first_stage = sm.OLS(t, X_first).fit()
        f_stat = float(first_stage.fvalue)

        z_crit = 1.959964
        return {
            "late": late_val,
            "se": se_val,
            "ci": (late_val - z_crit * se_val, late_val + z_crit * se_val),
            "f_stat": f_stat,
            "n": n,
            "method": "2SLS (statsmodels)",
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Wald estimator fallback (no covariate adjustment)
    cov_yz = np.cov(y, z)[0, 1]
    cov_tz = np.cov(t, z)[0, 1]
    if abs(cov_tz) < 1e-12:
        raise ValueError(
            "Instrument has near-zero covariance with treatment; LATE is not identified (weak instrument)."
        )
    late_val = float(cov_yz / cov_tz)

    # Delta-method SE for Wald estimator with binary instrument
    z_unique = np.unique(z)
    if len(z_unique) == 2:
        z0, z1 = sorted(z_unique)
        y_z1 = y[z == z1]
        y_z0 = y[z == z0]
        t_z1 = t[z == z1]
        t_z0 = t[z == z0]
        n1_z = len(y_z1)
        n0_z = len(y_z0)
        denom = float(t_z1.mean() - t_z0.mean())
        if abs(denom) < 1e-12:
            se_val = float("nan")
        else:
            # Variance of the ratio via delta method
            var_num = np.var(y_z1, ddof=1) / n1_z + np.var(y_z0, ddof=1) / n0_z
            se_val = float(np.sqrt(var_num) / abs(denom))
    else:
        se_val = float("nan")

    # First stage F-stat
    X_first = sm.add_constant(z)
    first_stage = sm.OLS(t, X_first).fit()
    f_stat = float(first_stage.fvalue)

    z_crit = 1.959964
    ci = (
        (late_val - z_crit * se_val, late_val + z_crit * se_val)
        if np.isfinite(se_val)
        else (float("nan"), float("nan"))
    )

    return {
        "late": late_val,
        "se": se_val,
        "ci": ci,
        "f_stat": f_stat,
        "n": n,
        "method": "Wald estimator (no covariate adjustment)",
    }


late = estimate_late


def cheatsheet() -> str:
    return "estimate_late({}) -> Local Average Treatment Effect (LATE) via instrumental varia"
