# morie.fn — function file (hadesllm/morie)
"""Partially Linear IV (PLIV) for LATE via DoubleML or 2SLS fallback."""

import warnings

import pandas as pd
import statsmodels.api as sm


def estimate_pliv(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    r"""
    Partially Linear IV (PLIV) for the Local Average Treatment Effect (LATE)
    using an instrumental variable via DoubleML.

    The PLIV model is:

    .. math::

        Y = \\theta_0 D + g_0(X) + \\varepsilon, \\quad
        D = m_0(X) + f_0(Z, X) + v

    where Z is the instrument, satisfying relevance and exclusion restriction.

    Falls back to two-stage least squares (2SLS) via statsmodels if
    DoubleML is unavailable.

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the endogenous treatment variable.
    :param outcome: Column name of the outcome variable.
    :param instrument: Column name of the instrument.
    :param covariates: List of exogenous covariate column names.
    :param n_folds: Cross-fitting folds (DoubleML path). Default 5.
    :param random_state: Random seed. Default 42.
    :return: dict with keys ``late``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``, ``method``.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning.
        Econometrics Journal, 21(1), C1-C68.
    Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of
        causal effects using instrumental variables. JASA, 91(434), 444-455.
    """
    required_cols = [treatment, outcome, instrument] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")

    df = data[[treatment, outcome, instrument] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    try:
        import doubleml as dml
        from sklearn.linear_model import RidgeCV

        dml_data = dml.DoubleMLData(
            df,
            y_col=outcome,
            d_cols=treatment,
            z_cols=instrument,
            x_cols=covariates,
        )
        ml_l = RidgeCV()
        ml_m = RidgeCV()
        ml_r = RidgeCV()

        pliv = dml.DoubleMLPLIV(
            dml_data,
            ml_l=ml_l,
            ml_m=ml_m,
            ml_r=ml_r,
            n_folds=n_folds,
            n_rep=1,
        )
        pliv.fit()

        late = float(pliv.coef[0])
        se = float(pliv.se[0])
        ci = pliv.confint(level=0.95)
        ci_lower = float(ci.iloc[0, 0])
        ci_upper = float(ci.iloc[0, 1])
        pval = float(pliv.pval[0])
        method = "DoubleML PLIV"

    except ImportError:
        warnings.warn(
            "doubleml not available; falling back to 2SLS via statsmodels.",
            ImportWarning,
            stacklevel=2,
        )
        # 2SLS: first stage D ~ Z + X, second stage Y ~ D_hat + X
        X_cols = covariates + [instrument]
        X_first = sm.add_constant(df[X_cols].astype(float))
        first_stage = sm.OLS(df[treatment].astype(float), X_first).fit()
        d_hat = first_stage.fittedvalues

        X_second = sm.add_constant(pd.concat([d_hat.rename("d_hat"), df[covariates].astype(float)], axis=1))
        second_stage = sm.OLS(df[outcome].astype(float), X_second).fit()

        late = float(second_stage.params["d_hat"])
        se = float(second_stage.bse["d_hat"])
        ci_lower = float(second_stage.conf_int().loc["d_hat", 0])
        ci_upper = float(second_stage.conf_int().loc["d_hat", 1])
        pval = float(second_stage.pvalues["d_hat"])
        method = "2SLS (statsmodels fallback)"

    return {
        "late": late,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pval": pval,
        "n_obs": n_obs,
        "method": method,
    }


pliv_fn = estimate_pliv


def cheatsheet() -> str:
    return "estimate_pliv({}) -> Partially Linear IV (PLIV) for LATE via DoubleML or 2SLS fal"
