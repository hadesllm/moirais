# moirais.fn — function file (hadesllm/moirais)
"""Cramer-Rao lower bound."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Difficult to see. Always in motion is the future."


def cramer_rao_lower_bound(fisher_info, **kwargs) -> DescriptiveResult:
    """Compute the Cramer-Rao lower bound from Fisher information.

    For a scalar parameter the CRLB is 1/I(theta).
    For a matrix the diagonal of the inverse gives per-parameter bounds.

    Parameters
    ----------
    fisher_info : float or array-like
        Fisher information (scalar or matrix).

    Returns
    -------
    DescriptiveResult
    """
    fi = np.asarray(fisher_info, dtype=float)
    if fi.ndim == 0:
        if fi == 0:
            raise ValueError("Fisher information must be non-zero.")
        crlb = float(1.0 / fi)
        return DescriptiveResult(
            name="cramer_rao_lower_bound",
            value=crlb,
            extra={"fisher_info": float(fi), "crlb": crlb},
        )
    inv_fi = np.linalg.inv(fi)
    crlb_diag = np.diag(inv_fi)
    return DescriptiveResult(
        name="cramer_rao_lower_bound",
        value=float(np.sum(crlb_diag)),
        extra={"fisher_info_matrix": fi.tolist(), "crlb_diag": crlb_diag.tolist()},
    )


crlb = cramer_rao_lower_bound


def cheatsheet() -> str:
    return "cramer_rao_lower_bound({}) -> Cramer-Rao lower bound."
