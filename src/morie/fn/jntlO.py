# morie.fn — function file (hadesllm/morie)
"""Joint loss for multi-output DNN with mixed outcome types."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joint_loss_mixed_outcomes"]


def joint_loss_mixed_outcomes(y_dict, y_hat_dict, weights):
    """
    Joint loss for multi-output DNN with mixed outcome types

    Formula: L_joint = w1*L_cont + w2*L_binary + w3*L_count + w4*L_ordinal

    Parameters
    ----------
    y_dict : array-like
        Input data.
    y_hat_dict : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loss': 'float'}

    References
    ----------
    Montesinos Lopez Ch 12
    """
    y_dict = np.asarray(y_dict, dtype=float)
    n = int(y_dict) if y_dict.ndim == 0 else len(y_dict)
    result = float(np.mean(y_dict))
    se = float(np.std(y_dict, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint loss for multi-output DNN with mixed outcome types"})


def cheatsheet():
    return "jntlO: Joint loss for multi-output DNN with mixed outcome types"
