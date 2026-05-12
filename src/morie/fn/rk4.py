# morie.fn -- function file (hadesllm/morie)
"""Fourth-order Runge-Kutta ODE solver."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Stay on target. -- Gold Five"


def runge_kutta4(f, y0, t_span, n_steps: int = 100, **kwargs) -> DescriptiveResult:
    r"""
    Solve an ODE system using the classic 4th-order Runge-Kutta method.

    For :math:`\\dot{y} = f(t, y)`:

    .. math::

        k_1 &= h\\,f(t_n, y_n) \\\\
        k_2 &= h\\,f(t_n + h/2, y_n + k_1/2) \\\\
        k_3 &= h\\,f(t_n + h/2, y_n + k_2/2) \\\\
        k_4 &= h\\,f(t_n + h, y_n + k_3) \\\\
        y_{n+1} &= y_n + \\tfrac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Local truncation error is :math:`O(h^5)`, global error :math:`O(h^4)`.

    :param f: Callable f(t, y) returning derivative(s).
    :param y0: Initial condition (scalar or array).
    :param t_span: Tuple (t0, tf) for start and end time.
    :param n_steps: Number of integration steps. Default 100.
    :return: DescriptiveResult with final value and full trajectory.
    :raises ValueError: If n_steps < 1 or t_span invalid.

    References
    ----------
    Butcher, J. C. (2016). *Numerical Methods for Ordinary Differential
    Equations* (3rd ed.). Wiley. doi:10.1002/9781119121534
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    t0, tf = float(t_span[0]), float(t_span[1])
    if tf <= t0:
        raise ValueError(f"t_span end ({tf}) must be > start ({t0}).")

    h = (tf - t0) / n_steps
    y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i]
        k1 = h * np.atleast_1d(f(ti, yi))
        k2 = h * np.atleast_1d(f(ti + h / 2, yi + k1 / 2))
        k3 = h * np.atleast_1d(f(ti + h / 2, yi + k2 / 2))
        k4 = h * np.atleast_1d(f(ti + h, yi + k3))
        y[i + 1] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    final = y[-1].squeeze()

    return DescriptiveResult(
        name="runge_kutta4",
        value=float(final) if final.ndim == 0 else float(final[0]),
        extra={
            "final_state": final,
            "time": t,
            "trajectory": y,
            "step_size": h,
            "n_steps": n_steps,
            "order": 4,
        },
    )


rk4 = runge_kutta4


def cheatsheet() -> str:
    return "runge_kutta4({}) -> Fourth-order Runge-Kutta ODE solver."
