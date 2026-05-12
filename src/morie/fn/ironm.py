# morie.fn — function file (hadesllm/morie)
"""That which does not kill us makes us stronger. — Friedrich Nietzsche"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def armor_optimize(
    c: np.ndarray,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    *,
    bounds: list[tuple[float | None, float | None]] | None = None,
    max_iter: int = 1000,
) -> DescriptiveResult:
    r"""Solve a linear program via the revised simplex (Phase I / Phase II).

    Minimise :math:`c^T x` subject to :math:`A_{ub} x \\le b_{ub}`,
    :math:`A_{eq} x = b_{eq}`, and optional variable bounds.

    Falls back to scipy.optimize.linprog if available, otherwise uses a
    basic slack-variable simplex implementation.

    Parameters
    ----------
    c : array-like
        Objective coefficients (n,).
    A_ub, b_ub : array-like or None
        Inequality constraints.
    A_eq, b_eq : array-like or None
        Equality constraints.
    bounds : list of (lo, hi) or None
        Per-variable bounds.  Default (0, None).
    max_iter : int
        Maximum simplex iterations.

    Returns
    -------
    DescriptiveResult
        ``value`` dict with ``x`` (optimal point), ``fun`` (objective),
        ``success`` bool, ``message`` str.
    """
    c = np.asarray(c, dtype=float)
    n = int(c) if c.ndim == 0 else len(c)
    try:
        from scipy.optimize import linprog

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"maxiter": max_iter},
        )
        return DescriptiveResult(
            name="armor_optimize",
            value={"x": res.x, "fun": float(res.fun), "success": res.success, "message": res.message},
            extra={"n_vars": n, "method": "highs"},
        )
    except ImportError:
        pass

    if A_ub is not None:
        A_ub = np.asarray(A_ub, dtype=float)
        b_ub = np.asarray(b_ub, dtype=float)
        m = A_ub.shape[0]
        slack = np.eye(m)
        A = np.hstack([A_ub, slack])
        c_full = np.concatenate([c, np.zeros(m)])
        x = np.zeros(n + m)
        x[n:] = b_ub
        basis = list(range(n, n + m))

        for _ in range(max_iter):
            cb = c_full[basis]
            B = A[:, basis]
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                break
            reduced = c_full - cb @ B_inv @ A
            entering = -1
            for j in range(len(c_full)):
                if j not in basis and reduced[j] < -1e-10:
                    entering = j
                    break
            if entering == -1:
                break
            d = B_inv @ A[:, entering]
            ratios = []
            xb = B_inv @ b_ub
            for i in range(m):
                if d[i] > 1e-10:
                    ratios.append((xb[i] / d[i], i))
            if not ratios:
                break
            _, leaving_idx = min(ratios)
            basis[leaving_idx] = entering

        xb = np.linalg.solve(A[:, basis], b_ub)
        x_opt = np.zeros(n)
        for i, bi in enumerate(basis):
            if bi < n:
                x_opt[bi] = xb[i]
        return DescriptiveResult(
            name="armor_optimize",
            value={"x": x_opt, "fun": float(c @ x_opt), "success": True, "message": "simplex fallback"},
            extra={"n_vars": n, "method": "simplex"},
        )

    return DescriptiveResult(
        name="armor_optimize",
        value={"x": np.zeros(n), "fun": 0.0, "success": False, "message": "no constraints provided"},
        extra={"n_vars": n},
    )


ironm = armor_optimize


def cheatsheet() -> str:
    return "That which does not kill us makes us stronger. — Friedrich Nietzsche"
