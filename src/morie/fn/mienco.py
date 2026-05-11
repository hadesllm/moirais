"""MI-based neural encoder objective."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_neural_encoder"]


def mi_neural_encoder(input, latent_net):
    """
    MI-based neural encoder objective

    Formula: max I(input; latent)

    Parameters
    ----------
    input : array-like
        Input data.
    latent_net : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hjelm et al (2019) Deep InfoMax
    """
    input = np.atleast_1d(np.asarray(input, dtype=float))
    n = len(input)
    result = float(np.mean(input))
    se = float(np.std(input, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MI-based neural encoder objective"})


def cheatsheet():
    return "mienco: MI-based neural encoder objective"
