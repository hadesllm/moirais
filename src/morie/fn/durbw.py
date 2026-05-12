# morie.fn -- function file (hadesllm/morie)
"""Durbin-Watson statistic for residual autocorrelation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Who's the more foolish: the fool, or the fool who follows him?"


def durbin_watson_stat_fn(residuals: np.ndarray) -> DescriptiveResult:
    r"""Compute the Durbin-Watson statistic.

    .. math::

        d = \\frac{\\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\\sum_{t=1}^{n} e_t^2}

    Values near 2 indicate no autocorrelation; near 0 positive, near 4 negative.

    :param residuals: 1-D residual signal.
    :return: DescriptiveResult with DW statistic.
    """
    residuals = np.asarray(residuals, dtype=float).ravel()
    diff = np.diff(residuals)
    ss_diff = float(np.sum(diff**2))
    ss_res = float(np.sum(residuals**2))
    dw = ss_diff / ss_res if ss_res > 0 else 0.0
    return DescriptiveResult(
        name="durbin_watson",
        value=dw,
        extra={"dw": dw, "n": len(residuals)},
    )


durbw = durbin_watson_stat_fn


def cheatsheet() -> str:
    return "durbin_watson_stat_fn({}) -> Durbin-Watson statistic for residual autocorrelation."
