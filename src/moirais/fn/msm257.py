"""Numbered display equation (1.222) from MVSML chapter 1.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_general_eq_1_222"]


def mvsml_general_eq_1_222(We, can, see, that, the, best):
    """
    Numbered display equation (1.222) from MVSML chapter 1.

    Formula: 247.6 We can see that the best combination of hyperparameters, in partition 2, of the inner cross-validation is combination 4 (with dropout = 0.05, units = 67, and epochs = 247.6) since it has the lowest MSE (0.971) in the testing set (validation set). Then with the optimal values of dropout, epochs, and neurons, the ANN was reﬁtted, but using the whole second outer training set, and then, with the ﬁnal trained ANN, the predictions for the outer testing set were obtained. Finally, the plot of the observed and predicted values of the outer testing set was done, as shown in Fig. 11.9. The MSE was calculated and was equal to (0.810), which is lower than the MSE obtained

    Parameters
    ----------
    We : array-like
        Input data.
    can : array-like
        Input data.
    see : array-like
        Input data.
    that : array-like
        Input data.
    the : array-like
        Input data.
    best : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (1.222) [Multivariate Statistical Machine Learnin [Pages 427-476] [2026-04-16].pdf]
    """
    We = np.atleast_1d(np.asarray(We, dtype=float))
    n = len(We)
    result = float(np.mean(We))
    se = float(np.std(We, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (1.222) from MVSML chapter 1."})


def cheatsheet():
    return "msm257: Numbered display equation (1.222) from MVSML chapter 1."
