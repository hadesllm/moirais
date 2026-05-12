# morie.fn -- function file (hadesllm/morie)
"""Newton-Raphson root finding."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. -- Socrates"


def newton_root(f, df, x0: float, tol: float = 1e-10, max_iter: int = 100, **kwargs) -> DescriptiveResult:
    """
    Find a root of f(x) = 0 using the Newton-Raphson method.

    Iteration:

    .. math::

        x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}

    Converges quadratically when :math:`f'(x^*) \\neq 0`.

    :param f: Callable f(x) whose root is sought.
    :param df: Callable f'(x), the derivative of f.
    :param x0: Initial guess.
    :param tol: Convergence tolerance on |f(x)|. Default 1e-10.
    :param max_iter: Maximum iterations. Default 100.
    :return: DescriptiveResult with root estimate.
    :raises ValueError: If derivative is zero during iteration.

    References
    ----------
    Burden, R. L. & Faires, J. D. (2011). *Numerical Analysis* (9th ed.).
    Brooks/Cole.
    """
    x = float(x0)
    history = [x]

    for i in range(max_iter):
        fx = float(f(x))
        if abs(fx) < tol:
            return DescriptiveResult(
                name="newton_root",
                value=x,
                extra={
                    "root": x,
                    "f_at_root": fx,
                    "iterations": i,
                    "converged": True,
                    "history": np.array(history),
                },
            )
        dfx = float(df(x))
        if abs(dfx) < 1e-300:
            raise ValueError(f"Derivative is zero at x={x}, iteration {i}.")
        x = x - fx / dfx
        history.append(x)

    fx = float(f(x))
    return DescriptiveResult(
        name="newton_root",
        value=x,
        extra={
            "root": x,
            "f_at_root": fx,
            "iterations": max_iter,
            "converged": abs(fx) < tol,
            "history": np.array(history),
        },
    )


newtn = newton_root


def cheatsheet() -> str:
    return "newton_root({}) -> Newton-Raphson root finding."
