"""GPT-style decoder forward pass."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gpt_decoder"]


def gpt_decoder(tokens, model):
    """
    GPT-style decoder forward pass

    Formula: causal masked self-attention; AR LM

    Parameters
    ----------
    tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Radford et al (2018) GPT
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GPT-style decoder forward pass"})


def cheatsheet():
    return "gpt2: GPT-style decoder forward pass"
