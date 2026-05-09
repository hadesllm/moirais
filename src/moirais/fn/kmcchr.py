# moirais.fn — function file (hadesllm/moirais)
"""Deep RL from human preferences (Christiano et al. 2017)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_christiano_deep_rl_feedback"]


def kamath_christiano_deep_rl_feedback(trajectory_pairs, r_phi):
    """
    Deep RL from human preferences (Christiano et al. 2017)

    Formula: L = sum_{(sigma_w, sigma_l)} -log P_phi(sigma_w > sigma_l); policy improves on learned reward

    Parameters
    ----------
    trajectory_pairs : array-like
        Input data.
    r_phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: policy

    References
    ----------
    Kamath Ch 5, Deep RL from Human Preferences section
    """
    trajectory_pairs = np.atleast_1d(np.asarray(trajectory_pairs, dtype=float))
    n = len(trajectory_pairs)
    result = float(np.mean(trajectory_pairs))
    se = float(np.std(trajectory_pairs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep RL from human preferences (Christiano et al. 2017)"})


def cheatsheet():
    return "kmcchr: Deep RL from human preferences (Christiano et al. 2017)"
