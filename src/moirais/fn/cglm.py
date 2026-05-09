# moirais.fn — function file (hadesllm/moirais)
"""Complex survey GLM with weights, clustering, and stratification."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def complex_survey_glm(
    df: pd.DataFrame,
    formula: str,
    weight_col: str,
    *,
    family: str = "gaussian",
    cluster_col: str | None = None,
    strata_col: str | None = None,
) -> object:
    """
    Fit a GLM with complex survey design (weights, optional clustering, strata).

    Model is fit using statsmodels WLS/GLM with analytic ``var_weights``
    (not freq_weights -- see :meth:`SurveyDesign.svyglm` for the rationale).

    When ``cluster_col`` is provided, cluster-robust (sandwich) standard errors
    are applied via ``fit(cov_type='cluster', cov_kwds={'groups': ...})``.

    Strata support is informational only at this level; stratum-specific
    estimates require :func:`subpopulation_estimate`.

    Supported families (string): ``"gaussian"``, ``"binomial"``, ``"poisson"``,
    ``"gamma"``, ``"negativebinomial"``.

    :param df: Input DataFrame.
    :param formula: Patsy-compatible formula string (e.g. ``"y ~ x1 + x2"``).
    :param weight_col: Column name of analytic/probability survey weights.
    :param family: GLM family string. Default ``"gaussian"``.
    :param cluster_col: Column identifying clusters for robust SE. Default None.
    :param strata_col: Column identifying strata (informational only). Default None.
    :return: Fitted statsmodels GLM result object.
    :raises ValueError: If required columns are missing or family is unrecognised.

    References
    ----------
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley.
    White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
        and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
    """
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame.")
    if cluster_col is not None and cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame.")
    if strata_col is not None and strata_col not in df.columns:
        raise ValueError(f"Strata column '{strata_col}' not found in DataFrame.")

    _family_map = {
        "gaussian": sm.families.Gaussian(),
        "binomial": sm.families.Binomial(),
        "poisson": sm.families.Poisson(),
        "gamma": sm.families.Gamma(),
        "negativebinomial": sm.families.NegativeBinomial(),
    }
    family_str = family.lower().replace("-", "").replace("_", "")
    if family_str not in _family_map:
        raise ValueError(f"Unknown family '{family}'. Choose from: {list(_family_map.keys())}.")
    family_obj = _family_map[family_str]

    w = df[weight_col].astype(float)
    if np.any(w.values <= 0):
        raise ValueError("All survey weights must be > 0.")

    model = smf.glm(formula=formula, data=df, family=family_obj, var_weights=w)

    if cluster_col is not None:
        groups = df[cluster_col].values
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        result = model.fit(cov_type="HC3")

    return result


cglm_fn = complex_survey_glm


def cheatsheet() -> str:
    return "complex_survey_glm({}) -> Complex survey GLM with weights, clustering, and stratificat"
