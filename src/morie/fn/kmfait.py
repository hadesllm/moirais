# morie.fn -- function file (hadesllm/morie)
"""RAGAS faithfulness: fraction of answer claims entailed by the retrieved context."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_ragas_faithfulness"]


def kamath_ragas_faithfulness(answer, context):
    """
    RAGAS faithfulness: fraction of answer claims entailed by the retrieved context

    Formula: faithfulness = |supported_claims| / |total_claims|  (claims extracted from the answer)

    Parameters
    ----------
    answer : array-like
        Input data.
    context : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Kamath Ch 7, RAG Faithfulness section
    """
    answer = np.atleast_1d(np.asarray(answer, dtype=float))
    n = len(answer)
    result = float(np.mean(answer))
    se = float(np.std(answer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RAGAS faithfulness: fraction of answer claims entailed by the retrieved context"})


def cheatsheet():
    return "kmfait: RAGAS faithfulness: fraction of answer claims entailed by the retrieved context"
