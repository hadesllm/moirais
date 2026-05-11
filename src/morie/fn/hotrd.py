# morie.fn — function file (hadesllm/morie)
"""Newton's method convergence. 'Faster! Faster!' -- Hot Rod"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def newton_convergence(
    f,
    fprime,
    x0: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    fprime2=None,
) -> DescriptiveResult:
    """Newton-Raphson root finding with convergence rate analysis.

    Tracks the convergence order by computing successive error ratios.

    Parameters
    ----------
    f : callable
        Function f(x) to find root of.
    fprime : callable
        First derivative f'(x).
    x0 : float
        Initial guess.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    fprime2 : callable, optional
        Second derivative for Halley's method variant.

    Returns
    -------
    DescriptiveResult
        With ``value`` = root estimate and ``extra`` containing
        iteration history and estimated convergence order.
    """
    x = float(x0)
    history = [x]
    errors = []

    for i in range(max_iter):
        fx = f(x)
        dfx = fprime(x)
        if abs(dfx) < 1e-30:
            break

        if fprime2 is not None:
            d2fx = fprime2(x)
            denom = 2 * dfx**2 - fx * d2fx
            if abs(denom) < 1e-30:
                step = fx / dfx
            else:
                step = 2 * fx * dfx / denom
        else:
            step = fx / dfx

        x_new = x - step
        history.append(x_new)
        errors.append(abs(x_new - x))

        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    conv_order = np.nan
    if len(errors) >= 3:
        ratios = []
        for j in range(1, len(errors) - 1):
            if errors[j - 1] > 1e-30 and errors[j] > 1e-30:
                r = np.log(max(errors[j + 1], 1e-300) / max(errors[j], 1e-300)) / np.log(
                    max(errors[j], 1e-300) / max(errors[j - 1], 1e-300)
                )
                if np.isfinite(r) and 0 < r < 10:
                    ratios.append(r)
        if ratios:
            conv_order = float(np.median(ratios))

    return DescriptiveResult(
        name="newton_convergence",
        value=float(x),
        extra={
            "n_iter": len(history) - 1,
            "residual": abs(f(x)),
            "convergence_order": conv_order,
            "history": np.array(history),
        },
    )


hotrd = newton_convergence


def cheatsheet() -> str:
    return "newton_convergence({}) -> Newton's method convergence. 'Faster! Faster!' -- Hot Rod"
