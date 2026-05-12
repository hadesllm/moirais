# morie.fn -- function file (hadesllm/morie)
"""Basic reproduction number R0."""

import numpy as np

from ._containers import ESRes


def basic_reproduction_number(
    beta: float | None = None,
    gamma: float | None = None,
    attack_rate: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> ESRes:
    """Compute the basic reproduction number R0.

    Method 1 (direct): R0 = beta / gamma.

    Method 2 (from attack rate): solve 1 - AR = exp(-R0 * AR) via Newton's
    method, where AR is the final attack rate (proportion infected).

    Parameters
    ----------
    beta : float or None
        Transmission rate.
    gamma : float or None
        Recovery rate.
    attack_rate : float or None
        Final attack rate (0, 1). Used when beta/gamma unknown.
    tol : float, default 1e-8
        Newton convergence tolerance.
    max_iter : int, default 100
        Maximum Newton iterations.

    Returns
    -------
    ESRes

    References
    ----------
    Dietz, K. (1993). The estimation of the basic reproduction number for
    infectious diseases. Statistical Methods in Medical Research, 2(1),
    23-41.
    """
    if beta is not None and gamma is not None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        r0_val = beta / gamma
        method = "direct"
    elif attack_rate is not None:
        if not 0 < attack_rate < 1:
            raise ValueError("attack_rate must be in (0, 1)")
        ar = attack_rate
        r0_val = 2.0
        for _ in range(max_iter):
            f = 1 - ar - np.exp(-r0_val * ar)
            fp = ar * np.exp(-r0_val * ar)
            if abs(fp) < 1e-15:
                break
            r0_val = r0_val - f / fp
            if abs(f) < tol:
                break
        method = "attack_rate_newton"
    else:
        raise ValueError("Provide (beta, gamma) or attack_rate")

    return ESRes(
        measure="R0",
        estimate=float(r0_val),
        extra={"method": method},
    )


r0 = basic_reproduction_number


def cheatsheet() -> str:
    return "basic_reproduction_number({}) -> Basic reproduction number R0."
