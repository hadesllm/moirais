"""Tangent space projection for semiparametric models.

Projects a score function onto the nuisance tangent space to compute
the efficient influence function as the residual. This is a
fundamental operation in semiparametric efficiency theory.

References
----------
Bickel, P. J., Klaassen, C. A. J., Ritov, Y., & Wellner, J. A. (1993).
*Efficient and Adaptive Estimation for Semiparametric Models*. Springer.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*. Springer. Chapters 2, 14.

van der Vaart, A. W. (1998). *Asymptotic Statistics*. Cambridge
University Press. Chapter 25.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def tgscr(
    score: np.ndarray,
    nuisance_scores: np.ndarray,
    *,
    regularize: float = 1e-8,
) -> dict[str, Any]:
    r"""Project a score onto the nuisance tangent space.

    Given a parametric score :math:`S_\theta` and a matrix of nuisance
    tangent directions :math:`\Lambda = [\lambda_1, \ldots, \lambda_k]`,
    the projection is:

    .. math::

        \Pi_\Lambda S_\theta = \Lambda
            (\Lambda^\top \Lambda)^{-1} \Lambda^\top S_\theta

    The efficient score is the residual:

    .. math::

        S_{\mathrm{eff}} = S_\theta - \Pi_\Lambda S_\theta

    This achieves the semiparametric information bound
    :math:`I_{\mathrm{eff}} = E[S_{\mathrm{eff}}^2]`.

    Parameters
    ----------
    score : np.ndarray
        Parametric score vector, shape ``(n,)``.
    nuisance_scores : np.ndarray
        Nuisance tangent directions, shape ``(n, k)`` where *k* is the
        dimension of the nuisance tangent space.
    regularize : float
        Ridge regularization for the Gram matrix inverse.

    Returns
    -------
    dict[str, Any]
        ``projection`` (array), ``efficient_score`` (array),
        ``info_bound``, ``info_loss_fraction``, ``n``, ``k``.
    """
    score = np.asarray(score, dtype=float)
    nuisance_scores = np.asarray(nuisance_scores, dtype=float)
    if nuisance_scores.ndim == 1:
        nuisance_scores = nuisance_scores[:, None]

    n, k = nuisance_scores.shape

    gram = nuisance_scores.T @ nuisance_scores / n
    gram += regularize * np.eye(k)

    cross = nuisance_scores.T @ score / n

    try:
        coeffs = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(gram, cross, rcond=None)[0]

    projection = nuisance_scores @ coeffs
    efficient_score = score - projection

    info_full = float(np.mean(score**2))
    info_eff = float(np.mean(efficient_score**2))
    info_loss = 1.0 - info_eff / max(info_full, 1e-12)

    return {
        "projection": projection,
        "efficient_score": efficient_score,
        "info_bound": info_eff,
        "info_loss_fraction": info_loss,
        "n": n,
        "k": k,
    }


tgscr_fn = tgscr


def cheatsheet() -> str:
    return "tgscr(score, nuisance_scores) -> Tangent space projection (Kosorok 2008, Ch. 2, 14)."
