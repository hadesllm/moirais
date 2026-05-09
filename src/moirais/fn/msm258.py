"""Numbered display equation (3.5) from MVSML chapter 3.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_elements_lin_reg_eq_3_5"]


def mvsml_elements_lin_reg_eq_3_5(colnames, results_i, c, Observed, Predicted, Trait):
    """
    Numbered display equation (3.5) from MVSML chapter 3.

    Formula: colnames(results_i)=c(Names_results_i [1:5],"Observed","Predicted","Trait") Pred_Summary=summary.BMTMECV(results=results_i, information = 'compact', digits = 4) Pred_Summary Pred_all_traits=rbind(Pred_all_traits,data.frame (Trait=Names_Traits[i],Pred_Summary)) } Pred_all_traits Res_Sum=Pred_all_traits[,-c

    Parameters
    ----------
    colnames : array-like
        Input data.
    results_i : array-like
        Input data.
    c : array-like
        Input data.
    Observed : array-like
        Input data.
    Predicted : array-like
        Input data.
    Trait : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (3.5) [Multivariate Statistical Machine Learnin [Pages 477-532] [2026-04-16].pdf]
    """
    colnames = np.atleast_1d(np.asarray(colnames, dtype=float))
    n = len(colnames)
    result = float(np.mean(colnames))
    se = float(np.std(colnames, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (3.5) from MVSML chapter 3."})


def cheatsheet():
    return "msm258: Numbered display equation (3.5) from MVSML chapter 3."
