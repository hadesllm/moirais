# morie.fn -- function file (hadesllm/morie)
"""Semiparametric efficiency bound (information bound)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EfficiencyBoundResult:
    """Result of semiparametric efficiency bound computation.

    Attributes
    ----------
    efficiency_bound : float
        The semiparametric efficiency bound (Cramer-Rao lower bound for
        the semiparametric model).
    fisher_info : float
        Estimated Fisher information for the parametric submodel.
    achieved_variance : float
        Variance of the provided estimator (if influence values given).
    efficiency_ratio : float
        Ratio of efficiency bound to achieved variance (1.0 = efficient).
    is_efficient : bool
        True if achieved_variance is within tolerance of the bound.
    n : int
        Sample size.
    """

    efficiency_bound: float
    fisher_info: float
    achieved_variance: float
    efficiency_ratio: float
    is_efficient: bool
    n: int


def effic(
    scores: np.ndarray,
    *,
    influence_values: np.ndarray | None = None,
    tol: float = 0.05,
) -> EfficiencyBoundResult:
    r"""
    Compute the semiparametric efficiency bound and assess estimator efficiency.

    In a semiparametric model :math:`\{P_{\theta, \eta} : \theta \in \Theta,
    \eta \in \mathcal{H}\}`, the efficiency bound for :math:`\theta` is:

    .. math::

        V_{\text{eff}} = \bigl(E[\tilde{S}_{\text{eff}}
        \tilde{S}_{\text{eff}}^\top]\bigr)^{-1}

    where :math:`\tilde{S}_{\text{eff}}` is the efficient score, the projection
    of the parametric score onto the orthocomplement of the nuisance tangent space.

    For a scalar parameter, the efficiency bound equals the inverse of the
    Fisher information along the efficient score direction:

    .. math::

        V_{\text{eff}} = \frac{1}{I_{\text{eff}}}
        = \frac{1}{E[\tilde{S}_{\text{eff}}^2]}

    An estimator is semiparametrically efficient if its influence function
    equals the efficient influence function, i.e., its asymptotic variance
    achieves the bound.

    :param scores: 1-D array of efficient score values evaluated at each
        observation, i.e., :math:`\tilde{S}_{\text{eff}}(X_i)`.
    :param influence_values: Optional 1-D array of influence function values
        from an estimator to assess. If None, only the bound is computed
        (achieved_variance set to bound, efficiency_ratio = 1.0).
    :param tol: Relative tolerance for declaring efficiency. Default 0.05
        (5% relative difference allowed).
    :return: EfficiencyBoundResult with bound, Fisher info, and efficiency flag.
    :raises ValueError: If scores is empty.

    References
    ----------
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*, Ch. 4 (Semiparametric efficiency bounds).
    Springer. DOI:10.1007/978-0-387-74978-5

    Bickel, P.J., Klaassen, C.A.J., Ritov, Y., & Wellner, J.A. (1993).
    *Efficient and Adaptive Estimation for Semiparametric Models*.
    Johns Hopkins University Press.

    Newey, W.K. (1990). Semiparametric efficiency bounds. *Journal of Applied
    Econometrics*, 5(2), 99--135.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("scores must be non-empty.")

    n = scores.size

    fisher_info = float(np.mean(scores**2))

    if fisher_info < 1e-15:
        eff_bound = np.inf
    else:
        eff_bound = 1.0 / fisher_info

    if influence_values is not None:
        influence_values = np.asarray(influence_values, dtype=float).ravel()
        if influence_values.size != n:
            raise ValueError(
                f"influence_values length ({influence_values.size}) must "
                f"match scores length ({n})."
            )
        achieved_var = float(np.var(influence_values, ddof=0))
    else:
        achieved_var = eff_bound

    if achieved_var < 1e-15:
        ratio = 1.0 if eff_bound < 1e-15 else 0.0
    else:
        ratio = eff_bound / achieved_var

    is_eff = abs(ratio - 1.0) <= tol

    return EfficiencyBoundResult(
        efficiency_bound=eff_bound,
        fisher_info=fisher_info,
        achieved_variance=achieved_var,
        efficiency_ratio=ratio,
        is_efficient=is_eff,
        n=n,
    )


def cheatsheet() -> str:
    return "effic({scores}) -> Semiparametric efficiency bound."
