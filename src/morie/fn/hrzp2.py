# morie.fn -- function file (hadesllm/morie)
"""Bandwidth selection for PLR (Horowitz 2009, Ch 3).

    h_opt = c * sigma_Z * n^{-1/5}

with c = 1.06 (Silverman) by default.
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_plr_bandwidth"]


def horowitz_plr_bandwidth(x, y, c=1.06):
    """Silverman-style PLR bandwidth.

    Parameters
    ----------
    x : array-like
        Nonparametric covariate Z (the kernel smooths on this).
    y : array-like
        Response (only used to verify length consistency).
    c : float, optional
        Multiplier (default 1.06, Silverman).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 5:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "plr-bandwidth (insufficient data)"})
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    sigma = min(s, iqr / 1.349) if iqr > 0 else s
    if sigma <= 0:
        sigma = max(s, 1e-6)
    h = c * sigma * n ** (-1.0 / 5.0)
    return RichResult(payload={
        "estimate": float(h), "n": n, "sigma": sigma, "c": float(c),
        "method": "Silverman rule h = c * sigma * n^(-1/5)",
    })


def cheatsheet():
    return "hrzp2: PLR bandwidth (Silverman ROT)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(3)
    z = rng.standard_normal(1000)
    res = horowitz_plr_bandwidth(z, z)
    print(res)
    assert 0.1 < res["estimate"] < 1.0
