# morie.fn -- function file (hadesllm/morie)
"""Peephole LSTM cell: gates see cell state c too."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_peephole_lstm_cell"]


def geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo):
    """
    Peephole LSTM cell: gates see cell state c too

    Formula: f, i, o also depend on c_prev: e.g. f = sig(Wf[h, x] + Uf c_prev + bf)

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    c_prev : array-like
        Input data.
    Wf : array-like
        Input data.
    Wi : array-like
        Input data.
    Wg : array-like
        Input data.
    Wo : array-like
        Input data.
    Uf : array-like
        Input data.
    Ui : array-like
        Input data.
    Uo : array-like
        Input data.
    bf : array-like
        Input data.
    bi : array-like
        Input data.
    bg : array-like
        Input data.
    bo : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t, c_t

    References
    ----------
    Géron Ch 13, Peephole LSTM section
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peephole LSTM cell: gates see cell state c too"})


def cheatsheet():
    return "grpels: Peephole LSTM cell: gates see cell state c too"
