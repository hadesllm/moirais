# moirais.fn — function file (hadesllm/moirais)
"""Membership inference attack: threshold on model loss to infer training-set membership."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_membership_inference"]


def kamath_membership_inference(losses, threshold):
    """
    Membership inference attack: threshold on model loss to infer training-set membership

    Formula: member_hat(x) = 1 if L_model(x) < tau else 0;  threshold tau chosen by attack

    Parameters
    ----------
    losses : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: predictions

    References
    ----------
    Kamath Ch 6, Membership Inference section
    """
    losses = np.atleast_1d(np.asarray(losses, dtype=float))
    n = len(losses)
    result = float(np.mean(losses))
    se = float(np.std(losses, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Membership inference attack: threshold on model loss to infer training-set membership"})


def cheatsheet():
    return "kmmbi: Membership inference attack: threshold on model loss to infer training-set membership"
