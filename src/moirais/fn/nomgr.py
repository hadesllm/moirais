# moirais.fn — function file (hadesllm/moirais)
"""Fagan nomogram (post-test probability). 'There is always a bigger fish.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def fagan_nomogram(
    prior: float,
    lr_pos: float,
) -> DescriptiveResult:
    """
    Compute post-test probability using Fagan's nomogram (Bayes' theorem).

    .. math::

        \\text{post-test odds} = \\text{pre-test odds} \\times LR^{+}

    .. math::

        P(D|T^{+}) = \\frac{\\text{post-test odds}}{1 + \\text{post-test odds}}

    :param prior: Pre-test probability (prevalence), in (0, 1).
    :param lr_pos: Positive likelihood ratio (LR+), must be > 0.
    :return: DescriptiveResult with post-test probability as value.
    :raises ValueError: If prior not in (0,1) or lr_pos <= 0.

    References
    ----------
    Fagan, T. J. (1975). Nomogram for Bayes theorem. New England Journal
    of Medicine, 293(5), 257. doi:10.1056/NEJM197507312930513
    """
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior must be in (0, 1), got {prior}.")
    if lr_pos <= 0.0:
        raise ValueError(f"lr_pos must be > 0, got {lr_pos}.")

    pre_odds = prior / (1.0 - prior)
    post_odds = pre_odds * lr_pos
    post_prob = post_odds / (1.0 + post_odds)

    return DescriptiveResult(
        name="Fagan Nomogram",
        value=float(np.round(post_prob, 6)),
        extra={
            "pre_test_probability": prior,
            "pre_test_odds": float(np.round(pre_odds, 4)),
            "post_test_odds": float(np.round(post_odds, 4)),
            "post_test_probability": float(np.round(post_prob, 6)),
            "lr_positive": lr_pos,
        },
    )


nomgr = fagan_nomogram


def cheatsheet() -> str:
    return "fagan_nomogram({}) -> Fagan nomogram (post-test probability). 'There is always a b"
