# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian FDR. 'Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bayesian_fdr(
    posterior_probs: np.ndarray,
    threshold: float = 0.5,
) -> DescriptiveResult:
    """Bayesian False Discovery Rate (Newton et al., 2004).

    Given posterior probabilities of the alternative hypothesis
    for each test, the Bayesian FDR is the average posterior
    probability of the null among those declared significant.

    :param posterior_probs: 1-D array of P(H1 | data) for each test.
    :param threshold: Minimum posterior probability to declare discovery.
    :return: DescriptiveResult with Bayesian FDR and discovery count.
    """
    pp = np.asarray(posterior_probs, dtype=float).ravel()
    discoveries = pp >= threshold
    n_disc = int(np.sum(discoveries))
    if n_disc == 0:
        bfdr = 0.0
    else:
        bfdr = float(np.mean(1.0 - pp[discoveries]))

    return DescriptiveResult(
        name="bfdr",
        value=bfdr,
        extra={
            "n_total": len(pp),
            "n_discoveries": n_disc,
            "threshold": threshold,
            "mean_posterior_h1": float(np.mean(pp)),
        },
    )


bfdr = bayesian_fdr


def cheatsheet() -> str:
    return "bayesian_fdr({}) -> Bayesian FDR. 'Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius'"
