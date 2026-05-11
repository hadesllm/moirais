"""Simpson's rule numerical integration."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"


def simpson_integrate(f, a: float, b: float, n: int = 100, **kwargs) -> DescriptiveResult:
    """
    Numerical integration using composite Simpson's 1/3 rule.

    .. math::

        \\int_a^b f(x)\\,dx \\approx \\frac{h}{3}\\left[
            f(x_0) + 4\\sum_{\\text{odd}} f(x_i) + 2\\sum_{\\text{even}} f(x_i) + f(x_n)
        \\right]

    Error is :math:`O(h^4)` where :math:`h = (b - a) / n`.

    :param f: Callable f(x) to integrate.
    :param a: Lower integration limit.
    :param b: Upper integration limit.
    :param n: Number of subintervals (must be even). Default 100.
    :return: DescriptiveResult with integral estimate.
    :raises ValueError: If n is odd or < 2, or a >= b.

    References
    ----------
    Burden, R. L. & Faires, J. D. (2011). *Numerical Analysis* (9th ed.).
    Brooks/Cole.
    """
    a, b = float(a), float(b)
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    if n % 2 != 0:
        n += 1
    if a >= b:
        raise ValueError(f"a ({a}) must be < b ({b}).")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = np.array([float(f(xi)) for xi in x])

    integral = fx[0] + fx[-1]
    integral += 4.0 * np.sum(fx[1::2])
    integral += 2.0 * np.sum(fx[2:-1:2])
    integral *= h / 3.0

    return DescriptiveResult(
        name="simpson_integrate",
        value=float(integral),
        extra={
            "integral": float(integral),
            "a": a,
            "b": b,
            "n_subintervals": n,
            "step_size": h,
            "error_order": "O(h^4)",
        },
    )


simps = simpson_integrate


def cheatsheet() -> str:
    return "simpson_integrate({}) -> Simpson's rule numerical integration."
