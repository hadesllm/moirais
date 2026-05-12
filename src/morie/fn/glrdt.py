# morie.fn -- function file (hadesllm/morie)
"""Generalized Likelihood Ratio change-point detector."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what we repeatedly do. Excellence is not an act, but a habit. -- Aristotle"


def glr_detector(x, window: int = 50, **kwargs) -> DescriptiveResult:
    """Detect change points using the Generalized Likelihood Ratio test.

    Slides a window and computes GLR statistic for a mean-shift model
    at each candidate change point.

    Parameters
    ----------
    x : array-like
        Input signal.
    window : int
        Minimum samples on each side of candidate change point.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    glr_stat = np.zeros(n)
    for k in range(window, n - window):
        x1 = x[:k]
        x2 = x[k:]
        n1, n2 = len(x1), len(x2)
        mu1, mu2 = np.mean(x1), np.mean(x2)
        mu_all = np.mean(x)
        var_all = np.var(x)
        if var_all < 1e-12:
            glr_stat[k] = 0.0
            continue
        ss_full = np.sum((x - mu_all) ** 2)
        ss_split = np.sum((x1 - mu1) ** 2) + np.sum((x2 - mu2) ** 2)
        if ss_split < 1e-12:
            glr_stat[k] = n * np.log(ss_full + 1e-12)
        else:
            glr_stat[k] = n * np.log(ss_full / ss_split)
    best_cp = int(np.argmax(glr_stat))
    return DescriptiveResult(
        name="glr_detector",
        value=float(glr_stat[best_cp]),
        extra={
            "change_point": best_cp,
            "glr_statistic": glr_stat,
            "window": window,
        },
    )


glrdt = glr_detector


def cheatsheet() -> str:
    return "glr_detector({}) -> Generalized Likelihood Ratio change-point detector."
