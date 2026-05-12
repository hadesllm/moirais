# morie.fn — function file (hadesllm/morie)
"""Partially Linear Regression (PLR) ATE via DoubleML."""

import pandas as pd


def estimate_plr(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    r"""
    Partially Linear Regression (PLR) ATE via DoubleML.

    The PLR model is:

    .. math::

        Y = \\theta_0 D + g_0(X) + \\varepsilon, \\quad
        D = m_0(X) + v

    where :math:`\\theta_0` is the ATE, :math:`g_0` and :math:`m_0` are
    nuisance functions estimated via cross-fitting, and the final
    :math:`\\theta_0` estimate is debiased.

    Nuisance learners default to ridge-regularised linear regression for
    both :math:`g_0` (outcome) and :math:`m_0` (treatment propensity / residual).

    :param data: DataFrame containing all required columns.
    :param treatment: Column name of the binary or continuous treatment variable.
    :param outcome: Column name of the outcome variable.
    :param covariates: List of covariate column names.
    :param n_folds: Number of cross-fitting folds. Default 5.
    :param random_state: Random seed for reproducibility. Default 42.
    :return: dict with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``pval``, ``n_obs``.
    :raises ImportError: If ``doubleml`` is not installed.
    :raises ValueError: If required columns are missing.

    References
    ----------
    Chernozhukov et al. (2018). Double/debiased machine learning for treatment
        and structural parameters. Econometrics Journal, 21(1), C1-C68.
    Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML:
        An object-oriented implementation of double machine learning in Python.
        Journal of Machine Learning Research, 23(53), 1-6.
    """
    try:
        import doubleml as dml
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise ImportError(
            "doubleml and scikit-learn are required for estimate_plr. Install with: pip install doubleml scikit-learn"
        ) from exc

    required_cols = [treatment, outcome] + covariates
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")

    df = data[[treatment, outcome] + covariates].dropna().reset_index(drop=True)
    n_obs = len(df)

    dml_data = dml.DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=covariates)

    ml_l = RidgeCV()
    ml_m = RidgeCV()

    plr = dml.DoubleMLPLR(
        dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=1,
    )
    plr.fit(store_predictions=False)

    ate = float(plr.coef[0])
    se = float(plr.se[0])
    ci = plr.confint(level=0.95)
    ci_lower = float(ci.iloc[0, 0])
    ci_upper = float(ci.iloc[0, 1])
    pval = float(plr.pval[0])

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pval": pval,
        "n_obs": n_obs,
    }


plr_fn = estimate_plr


def cheatsheet() -> str:
    return "estimate_plr({}) -> Partially Linear Regression (PLR) ATE via DoubleML."
