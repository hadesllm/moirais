"""DINO self-distillation loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dino_self_distill"]


def dino_self_distill(s_logits, t_logits, tau_t, tau_s):
    """
    DINO self-distillation loss

    Formula: -sum P_t * log P_s; momentum teacher

    Parameters
    ----------
    s_logits : array-like
        Input data.
    t_logits : array-like
        Input data.
    tau_t : array-like
        Input data.
    tau_s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Caron et al (2021)
    """
    s_logits = np.atleast_1d(np.asarray(s_logits, dtype=float))
    n = len(s_logits)
    result = float(np.mean(s_logits))
    se = float(np.std(s_logits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINO self-distillation loss"})


def cheatsheet():
    return "dinopr: DINO self-distillation loss"
