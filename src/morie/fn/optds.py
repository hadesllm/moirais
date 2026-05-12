# morie.fn -- function file (hadesllm/morie)
"""D-optimal experimental design via coordinate exchange."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def optimal_design(
    n_points: int,
    n_factors: int,
    criterion: str = "D",
    seed: int = 42,
    max_iter: int = 100,
) -> DescriptiveResult:
    """Generate an approximately D-optimal design via coordinate exchange.

    The D-optimality criterion maximizes :math:`|X^T X|`, which
    minimizes the volume of the confidence ellipsoid for the
    parameter estimates.

    Parameters
    ----------
    n_points : int
        Number of design points (runs).
    n_factors : int
        Number of factors.
    criterion : str, default "D"
        Optimality criterion.  Currently only ``"D"`` is supported.
    seed : int, default 42
        Random seed.
    max_iter : int, default 100
        Maximum coordinate-exchange iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` is the log-determinant of :math:`X^T X`.
        ``extra`` has ``design`` (n_points x n_factors matrix on
        [-1, 1]) and ``n_iter``.

    References
    ----------
    Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange
    algorithm for constructing exact optimal experimental designs.
    *Technometrics*, 37(1), 60--69.

    Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
    *Optimum Experimental Designs, with SAS*. Oxford University Press.
    """
    if n_points < n_factors + 1:
        raise ValueError(f"n_points ({n_points}) must be >= n_factors+1 ({n_factors + 1}).")
    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors}.")
    if criterion.upper() != "D":
        raise ValueError(f"Only 'D' criterion supported, got '{criterion}'.")

    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 1.0], size=(n_points, n_factors))

    def _add_intercept(M: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(M.shape[0]), M])

    Xa = _add_intercept(X)
    best_det = np.linalg.det(Xa.T @ Xa)
    n_iter = 0

    for it in range(max_iter):
        improved = False
        for i in range(n_points):
            for j in range(n_factors):
                old_val = X[i, j]
                for candidate in [-1.0, 0.0, 1.0]:
                    if candidate == old_val:
                        continue
                    X[i, j] = candidate
                    Xa = _add_intercept(X)
                    new_det = np.linalg.det(Xa.T @ Xa)
                    if new_det > best_det:
                        best_det = new_det
                        improved = True
                    else:
                        X[i, j] = old_val
        n_iter = it + 1
        if not improved:
            break

    log_det = float(np.log(best_det)) if best_det > 0 else -np.inf

    return DescriptiveResult(
        name="DOptimalDesign",
        value=log_det,
        extra={
            "design": X.copy(),
            "n_iter": n_iter,
            "n_points": n_points,
            "n_factors": n_factors,
            "det_XtX": float(best_det),
        },
    )


optds = optimal_design


def cheatsheet() -> str:
    return "optimal_design({}) -> D-optimal experimental design via coordinate exchange."
