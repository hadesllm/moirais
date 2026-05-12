# morie.fn -- function file (hadesllm/morie)
"""Numerical differentiation via finite differences."""


from ._containers import DescriptiveResult

_QUOTE = "Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius"


def finite_diff(f, x: float, h: float = 1e-5, order: int = 1, method: str = "central", **kwargs) -> DescriptiveResult:
    """
    Numerical differentiation using finite difference formulas.

    - **central** (default): :math:`f'(x) \\approx \\frac{f(x+h) - f(x-h)}{2h}`,
      error :math:`O(h^2)`
    - **forward**: :math:`f'(x) \\approx \\frac{f(x+h) - f(x)}{h}`,
      error :math:`O(h)`
    - **backward**: :math:`f'(x) \\approx \\frac{f(x) - f(x-h)}{h}`,
      error :math:`O(h)`

    For order 2: :math:`f''(x) \\approx \\frac{f(x+h) - 2f(x) + f(x-h)}{h^2}`

    :param f: Callable f(x) to differentiate.
    :param x: Point at which to evaluate the derivative.
    :param h: Step size. Default 1e-5.
    :param order: Derivative order (1 or 2). Default 1.
    :param method: ``"central"``, ``"forward"``, or ``"backward"``. Default ``"central"``.
    :return: DescriptiveResult with derivative estimate.
    :raises ValueError: If order not in {1, 2} or h <= 0.

    References
    ----------
    Burden, R. L. & Faires, J. D. (2011). *Numerical Analysis* (9th ed.).
    Brooks/Cole.
    """
    x = float(x)
    h = float(h)
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}.")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}.")

    if order == 1:
        if method == "central":
            deriv = (f(x + h) - f(x - h)) / (2.0 * h)
            error_order = "O(h^2)"
        elif method == "forward":
            deriv = (f(x + h) - f(x)) / h
            error_order = "O(h)"
        elif method == "backward":
            deriv = (f(x) - f(x - h)) / h
            error_order = "O(h)"
        else:
            raise ValueError(f"method must be 'central', 'forward', or 'backward', got '{method}'.")
    else:
        deriv = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
        error_order = "O(h^2)"

    return DescriptiveResult(
        name="finite_diff",
        value=float(deriv),
        extra={
            "derivative": float(deriv),
            "x": x,
            "h": h,
            "order": order,
            "method": method,
            "error_order": error_order,
        },
    )


fdiff = finite_diff


def cheatsheet() -> str:
    return "finite_diff({}) -> Numerical differentiation via finite differences."
