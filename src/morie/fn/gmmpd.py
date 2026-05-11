# morie.fn — function file (hadesllm/morie)
"""Gaussian Mixture Model PDF evaluation."""

import numpy as np

from ._containers import DescriptiveResult
def gmm_pdf(x, means, covs, weights, **kwargs) -> DescriptiveResult:
    """
    Evaluate a Gaussian Mixture Model PDF at points *x*.

    .. math::

        p(x) = \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(x; \\mu_k, \\sigma_k^2)

    :param x: array-like of evaluation points.
    :param means: array-like of component means (K,).
    :param covs: array-like of component variances (K,).
    :param weights: array-like of mixture weights (K,), must sum to 1.
    :return: DescriptiveResult with log-likelihood as value and per-point PDF.

    References
    ----------
    McLachlan GJ, Peel D (2000). Finite Mixture Models.
    Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    means = np.asarray(means, dtype=np.float64).ravel()
    covs = np.asarray(covs, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    k = len(means)
    if len(covs) != k or len(weights) != k:
        raise ValueError("means, covs, and weights must have same length.")
    if np.any(covs <= 0):
        raise ValueError("All variances must be > 0.")
    weights = weights / weights.sum()
    pdf = np.zeros_like(x)
    for i in range(k):
        pdf += weights[i] * (1.0 / np.sqrt(2 * np.pi * covs[i]) * np.exp(-0.5 * ((x - means[i]) ** 2) / covs[i]))
    ll = float(np.sum(np.log(pdf + 1e-300)))
    return DescriptiveResult(
        name="gmm_pdf",
        value=ll,
        extra={"pdf": pdf.tolist(), "n_components": k, "n": len(x)},
    )


gmmpd = gmm_pdf


def cheatsheet() -> str:
    return "gmm_pdf({}) -> Gaussian Mixture Model PDF evaluation."
