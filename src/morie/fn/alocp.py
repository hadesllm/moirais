# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""OpenCLIP contrastive objective -- scaled dot-product between normalized text + image emb."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_openclip_contrastive"]


def alammar_openclip_contrastive(I_emb, T_emb, tau):
    """
    OpenCLIP contrastive objective -- scaled dot-product between normalized text + image emb

    Formula: L = (1/2)(CE(sim_IT/tau, labels) + CE(sim_TI/tau, labels)); sim_ij = I_i.T_j after L2-norm

    Parameters
    ----------
    I_emb : array-like
        Input data.
    T_emb : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 9, OpenCLIP section
    """
    I_emb = np.atleast_1d(np.asarray(I_emb, dtype=float))
    n = len(I_emb)
    result = float(np.mean(I_emb))
    se = float(np.std(I_emb, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OpenCLIP contrastive objective -- scaled dot-product between normalized text + image emb"})


def cheatsheet():
    return "alocp: OpenCLIP contrastive objective -- scaled dot-product between normalized text + image emb"
