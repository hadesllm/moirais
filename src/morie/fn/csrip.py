# morie.fn — function file (hadesllm/morie)
"""Restricted Isometry Property (RIP) check."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "This is the way. -- Din Djarin"


def rip_check(
    A,
    s: int,
    delta: float = 0.1,
    n_trials: int = 1000,
    seed: int = 42,
    **kwargs,
) -> DescriptiveResult:
    """
    Empirically estimate the Restricted Isometry Constant (RIC) of a matrix.

    A matrix satisfies the RIP of order *s* with constant :math:`\\delta_s`
    if for all *s*-sparse vectors *x*:

    .. math::

        (1 - \\delta_s) \\|x\\|_2^2 \\leq \\|Ax\\|_2^2 \\leq (1 + \\delta_s) \\|x\\|_2^2

    Estimated by random sampling of s-sparse unit vectors.

    :param A: (m, n) sensing matrix (should have unit-norm columns).
    :param s: Sparsity level to test.
    :param delta: Threshold for RIP satisfaction. Default 0.1.
    :param n_trials: Number of random s-sparse vectors to test. Default 1000.
    :param seed: Random seed. Default 42.
    :return: DescriptiveResult with estimated RIC and pass/fail.
    :raises ValueError: If s > n or s < 1.

    References
    ----------
    Candes, E. J. & Tao, T. (2005). Decoding by linear programming.
    *IEEE Trans. Inform. Theory*, 51(12), 4203-4215.
    Baraniuk, R. G. et al. (2008). A simple proof of the restricted
    isometry property for random matrices. *Constructive Approximation*,
    28(3), 253-263.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if s < 1 or s > n:
        raise ValueError(f"s must be in [1, {n}], got {s}.")

    rng = np.random.default_rng(seed)
    max_ratio = 0.0
    min_ratio = float("inf")

    for _ in range(n_trials):
        support = rng.choice(n, size=s, replace=False)
        x = np.zeros(n)
        x[support] = rng.standard_normal(s)
        x_norm_sq = float(np.dot(x, x))
        if x_norm_sq < 1e-300:
            continue
        ax_norm_sq = float(np.dot(A @ x, A @ x))
        ratio = ax_norm_sq / x_norm_sq
        max_ratio = max(max_ratio, ratio)
        min_ratio = min(min_ratio, ratio)

    delta_s = max(abs(max_ratio - 1.0), abs(min_ratio - 1.0))
    satisfies_rip = bool(delta_s <= delta)

    return DescriptiveResult(
        name="rip_check",
        value=delta_s,
        extra={
            "delta_s": delta_s,
            "satisfies_rip": satisfies_rip,
            "threshold": delta,
            "sparsity": s,
            "max_ratio": max_ratio,
            "min_ratio": min_ratio,
            "n_trials": n_trials,
            "matrix_shape": (m, n),
        },
    )


csrip = rip_check


def cheatsheet() -> str:
    return "rip_check({}) -> Restricted Isometry Property (RIP) check."
