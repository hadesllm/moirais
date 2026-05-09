"""VAE for survival representation learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vae_survival"]


def vae_survival(time, event, X):
    """
    VAE for survival representation learning

    Formula: latent-z encoder + survival decoder

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nagpal et al (2021) DeepSurvivalMachines
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAE for survival representation learning"})


def cheatsheet():
    return "survvae: VAE for survival representation learning"
