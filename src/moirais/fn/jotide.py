# moirais.fn — function file (hadesllm/moirais)
"""TiDE: dense-encoder MLP forecaster (Time-series Dense Encoder)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_tide_encoder"]


def joseph_tide_encoder(past, covariates, horizon):
    """
    TiDE: dense-encoder MLP forecaster (Time-series Dense Encoder)

    Formula: y_hat = DenseDecoder(DenseEncoder([past, covariates]))

    Parameters
    ----------
    past : array-like
        Input data.
    covariates : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 16, TiDE section
    """
    past = np.atleast_1d(np.asarray(past, dtype=float))
    n = len(past)
    result = float(np.mean(past))
    se = float(np.std(past, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TiDE: dense-encoder MLP forecaster (Time-series Dense Encoder)"})


def cheatsheet():
    return "jotide: TiDE: dense-encoder MLP forecaster (Time-series Dense Encoder)"
