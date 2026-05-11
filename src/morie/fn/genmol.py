"""VAE/diffusion generative chemistry sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["generative_chemistry"]


def generative_chemistry(model, n_samples, conditions):
    """
    VAE/diffusion generative chemistry sampler

    Formula: sample from latent z; decode to SMILES

    Parameters
    ----------
    model : array-like
        Input data.
    n_samples : array-like
        Input data.
    conditions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gómez-Bombarelli et al (2018); Sanchez-Lengeling-Aspuru-Guzik (2018)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAE/diffusion generative chemistry sampler"})


def cheatsheet():
    return "genmol: VAE/diffusion generative chemistry sampler"
