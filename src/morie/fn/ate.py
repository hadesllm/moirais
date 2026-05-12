# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""IPW-weighted OLS ATE estimator."""

import pandas as pd
import statsmodels.formula.api as smf


def estimate_ate(data: pd.DataFrame, outcome: str, treatment: str, weights_col: str) -> tuple[float, float]:
    """
    Estimate Average Treatment Effect (ATE) using a weighted linear model.

    :param data: The pandas DataFrame containing the analytical sample.
    :type data: pandas.DataFrame
    :param outcome: The name of the outcome variable column.
    :type outcome: str
    :param treatment: The name of the binary treatment indicator column.
    :type treatment: str
    :param weights_col: The name of the column containing the analytical weights (e.g. IPTW).
    :type weights_col: str
    :return: A tuple containing the estimated ATE coefficient and its standard error.
    :rtype: tuple[float, float]
    """
    formula = f"{outcome} ~ {treatment}"
    # HC3 robust covariance: corrects for heteroskedasticity introduced by
    # unequal IPTW weights.  Plain OLS/WLS SEs are downward-biased when
    # observation weights vary widely, producing anti-conservative inference.
    model = smf.wls(formula=formula, data=data, weights=data[weights_col]).fit(cov_type="HC3")
    return float(model.params[treatment]), float(model.bse[treatment])


ate_fn = estimate_ate


def cheatsheet() -> str:
    return "estimate_ate({}) -> IPW-weighted OLS ATE estimator."
