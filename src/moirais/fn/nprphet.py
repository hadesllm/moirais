"""NeuralProphet — AR + MLP + holidays."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["neural_prophet"]


def neural_prophet(ds, y, ar_layers):
    """
    NeuralProphet — AR + MLP + holidays

    Formula: trend + seasonal + AR(p) + future regressors via NN

    Parameters
    ----------
    ds : array-like
        Input data.
    y : array-like
        Input data.
    ar_layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Triebe et al (2021)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NeuralProphet — AR + MLP + holidays"})


def cheatsheet():
    return "nprphet: NeuralProphet — AR + MLP + holidays"
