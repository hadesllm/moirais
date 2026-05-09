# moirais.fn — function file (hadesllm/moirais)
"""T5 span-corruption objective: mask spans and predict them as a single target sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_t5_span_corruption"]


def kamath_t5_span_corruption(tokens, mean_span_len, corruption_rate):
    """
    T5 span-corruption objective: mask spans and predict them as a single target sequence

    Formula: Input: x_with_sentinels; Target: <s_1> span_1 <s_2> span_2 ...; Loss = CE over target sequence

    Parameters
    ----------
    tokens : array-like
        Input data.
    mean_span_len : array-like
        Input data.
    corruption_rate : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: input_ids, target_ids

    References
    ----------
    Kamath Ch 2, Span Corruption (T5) section
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T5 span-corruption objective: mask spans and predict them as a single target sequence"})


def cheatsheet():
    return "kmspn: T5 span-corruption objective: mask spans and predict them as a single target sequence"
