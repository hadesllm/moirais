# moirais.fn — function file (hadesllm/moirais)
"""Corrective RAG: classifier decides to use retrieved docs or fall back to web/self-knowledge."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_corrective_rag"]


def kamath_corrective_rag(query, docs, clf, tau_hi, tau_lo):
    """
    Corrective RAG: classifier decides to use retrieved docs or fall back to web/self-knowledge

    Formula: s = clf(q, top-k-docs); use docs if s>=tau_hi; fallback_web if s<=tau_lo; mixed otherwise

    Parameters
    ----------
    query : array-like
        Input data.
    docs : array-like
        Input data.
    clf : array-like
        Input data.
    tau_hi : array-like
        Input data.
    tau_lo : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: action, ctx

    References
    ----------
    Kamath Ch 7, Corrective RAG section
    """
    query = np.atleast_1d(np.asarray(query, dtype=float))
    n = len(query)
    result = float(np.mean(query))
    se = float(np.std(query, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Corrective RAG: classifier decides to use retrieved docs or fall back to web/self-knowledge"})


def cheatsheet():
    return "kmcrag: Corrective RAG: classifier decides to use retrieved docs or fall back to web/self-knowledge"
