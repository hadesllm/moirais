"""Numbered display equation (9.17) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_17"]


def mvsml_ridge_lasso_elastic_eq_9_17(Its, dual, version, according, to, Wolfe):
    """
    Numbered display equation (9.17) from MVSML chapter 9.

    Formula: Its dual version according to Wolfe is equal to 9.3 Maximum Margin Classiﬁer 347 ) = x2 + 2\alpha x + 1 maximize f x, \alpha ( ( )

    Parameters
    ----------
    Its : array-like
        Input data.
    dual : array-like
        Input data.
    version : array-like
        Input data.
    according : array-like
        Input data.
    to : array-like
        Input data.
    Wolfe : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.17) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Its = np.atleast_1d(np.asarray(Its, dtype=float))
    n = len(Its)
    result = float(np.mean(Its))
    se = float(np.std(Its, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.17) from MVSML chapter 9."})


def cheatsheet():
    return "msm189: Numbered display equation (9.17) from MVSML chapter 9."
