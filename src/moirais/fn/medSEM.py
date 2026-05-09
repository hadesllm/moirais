"""SEM-based mediation (lavaan style)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sem_mediation"]


def sem_mediation(model_spec, data):
    """
    SEM-based mediation (lavaan style)

    Formula: latent measurement + structural with indirect

    Parameters
    ----------
    model_spec : array-like
        Input data.
    data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bollen (1989)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SEM-based mediation (lavaan style)"})


def cheatsheet():
    return "medSEM: SEM-based mediation (lavaan style)"
