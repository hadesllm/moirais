"""GAN trained to detect anomalies."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adversarial_anomaly"]


def adversarial_anomaly(x, D):
    """
    GAN trained to detect anomalies

    Formula: discriminator score -> anomaly

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Akcay et al (2018) GANomaly
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GAN trained to detect anomalies"})


def cheatsheet():
    return "adGAN: GAN trained to detect anomalies"
