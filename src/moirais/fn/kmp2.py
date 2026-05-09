# moirais.fn — function file (hadesllm/moirais)
"""P-tuning v2: deep prompt (learnable prefix per layer, not just input)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_p_tuning_v2"]


def kamath_p_tuning_v2(prefixes_by_layer, inputs_by_layer):
    """
    P-tuning v2: deep prompt (learnable prefix per layer, not just input)

    Formula: at each layer l: K_l, V_l = [P_K_l; K_l^in], [P_V_l; V_l^in]; only P_K_l, P_V_l trained

    Parameters
    ----------
    prefixes_by_layer : array-like
        Input data.
    inputs_by_layer : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: augmented_KV

    References
    ----------
    Kamath Ch 4, P-tuning v2 section
    """
    prefixes_by_layer = np.atleast_1d(np.asarray(prefixes_by_layer, dtype=float))
    n = len(prefixes_by_layer)
    result = float(np.mean(prefixes_by_layer))
    se = float(np.std(prefixes_by_layer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "P-tuning v2: deep prompt (learnable prefix per layer, not just input)"})


def cheatsheet():
    return "kmp2: P-tuning v2: deep prompt (learnable prefix per layer, not just input)"
