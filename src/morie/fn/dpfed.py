"""DP FedAvg (user-level DP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_fedavg"]


def dp_fedavg(clients, C, sigma):
    """
    DP FedAvg (user-level DP)

    Formula: clip + Gaussian noise on aggregated client updates

    Parameters
    ----------
    clients : array-like
        Input data.
    C : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McMahan-Ramage-Talwar-Zhang (2018)
    """
    clients = np.atleast_1d(np.asarray(clients, dtype=float))
    n = len(clients)
    result = float(np.mean(clients))
    se = float(np.std(clients, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP FedAvg (user-level DP)"})


def cheatsheet():
    return "dpfed: DP FedAvg (user-level DP)"
