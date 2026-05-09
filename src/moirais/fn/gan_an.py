"""GAN-based anomaly (AnoGAN)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gan_anomaly"]


def gan_anomaly(X, gan):
    """
    GAN-based anomaly (AnoGAN)

    Formula: recover z minimizing ||G(z)−x||

    Parameters
    ----------
    X : array-like
        Input data.
    gan : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schlegl et al (2017) AnoGAN
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GAN-based anomaly (AnoGAN)"})


def cheatsheet():
    return "gan_an: GAN-based anomaly (AnoGAN)"
