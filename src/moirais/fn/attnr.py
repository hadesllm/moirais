# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Disattenuation (correction for attenuation). 'Size matters not.'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def attenuation_ratio(
    r_obs: float,
    rel_x: float,
    rel_y: float = 1.0,
) -> DescriptiveResult:
    """
    Correct an observed correlation for measurement unreliability.

    .. math::

        r_{\\text{true}} = \\frac{r_{xy}}{\\sqrt{\\rho_{xx'} \\, \\rho_{yy'}}}

    :param r_obs: Observed correlation coefficient.
    :param rel_x: Reliability of measure *x* (e.g. Cronbach's alpha), in (0, 1].
    :param rel_y: Reliability of measure *y*. Default 1.0 (criterion).
    :return: DescriptiveResult with disattenuated correlation.
    :raises ValueError: If reliabilities are not in (0, 1].

    References
    ----------
    Spearman, C. (1904). The proof and measurement of association between
    two things. American Journal of Psychology, 15(1), 72--101.
    """
    if not 0.0 < rel_x <= 1.0:
        raise ValueError(f"rel_x must be in (0, 1], got {rel_x}.")
    if not 0.0 < rel_y <= 1.0:
        raise ValueError(f"rel_y must be in (0, 1], got {rel_y}.")
    if not -1.0 <= r_obs <= 1.0:
        raise ValueError(f"r_obs must be in [-1, 1], got {r_obs}.")

    denom = np.sqrt(rel_x * rel_y)
    r_true = r_obs / denom

    return DescriptiveResult(
        name="Disattenuated Correlation",
        value=float(np.round(r_true, 6)),
        extra={
            "r_observed": r_obs,
            "r_disattenuated": float(np.round(r_true, 6)),
            "reliability_x": rel_x,
            "reliability_y": rel_y,
            "attenuation_factor": float(np.round(denom, 6)),
        },
    )


attnr = attenuation_ratio


def cheatsheet() -> str:
    return "attenuation_ratio({}) -> Disattenuation (correction for attenuation). 'Size matters n"
