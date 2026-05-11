"""Transfer function step and impulse response."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Study the past if you would define the future. — Confucius"


def transfer_function(
    num,
    den,
    t=None,
    n_points: int = 500,
    response: str = "step",
    **kwargs,
) -> DescriptiveResult:
    """
    Compute the step or impulse response of a continuous transfer function.

    Given :math:`H(s) = N(s) / D(s)`, simulates the output using partial
    fraction expansion and inverse Laplace via pole-residue form.

    For a first-order system :math:`H(s) = K / (\\tau s + 1)`, the step
    response is :math:`y(t) = K(1 - e^{-t/\\tau})`.

    :param num: Numerator polynomial coefficients (highest power first).
    :param den: Denominator polynomial coefficients (highest power first).
    :param t: Time vector. If None, auto-generated from 0 to 10.
    :param n_points: Number of time points if t is None. Default 500.
    :param response: ``"step"`` or ``"impulse"``. Default ``"step"``.
    :return: DescriptiveResult with DC gain, poles, and time response.
    :raises ValueError: If den is all zeros or response type invalid.

    References
    ----------
    Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.
    """
    num = np.atleast_1d(np.asarray(num, dtype=np.float64))
    den = np.atleast_1d(np.asarray(den, dtype=np.float64))

    if np.all(den == 0):
        raise ValueError("Denominator coefficients cannot all be zero.")
    if response not in ("step", "impulse"):
        raise ValueError(f"response must be 'step' or 'impulse', got '{response}'.")

    if t is None:
        t = np.linspace(0, 10, n_points)
    else:
        t = np.asarray(t, dtype=np.float64)

    poles = np.roots(den)
    dc_gain = float(np.polyval(num, 0) / np.polyval(den, 0)) if np.polyval(den, 0) != 0 else float("inf")

    if response == "step":
        num_s = np.polymul(num, [1.0])
        den_s = np.polymul(den, [1.0, 0.0])
    else:
        num_s = num.copy()
        den_s = den.copy()

    r, p, k = _partial_fraction(num_s, den_s)

    y = np.zeros_like(t, dtype=np.complex128)
    for ri, pi in zip(r, p):
        y += ri * np.exp(pi * t)
    for i, ki in enumerate(k):
        y += ki * t**i
    y = np.real(y)

    return DescriptiveResult(
        name="transfer_function",
        value=dc_gain,
        extra={
            "dc_gain": dc_gain,
            "poles": poles,
            "time": t,
            "output": y,
            "response_type": response,
            "num": num,
            "den": den,
        },
    )


def _partial_fraction(num, den):
    """Partial fraction expansion using numpy polynomial division."""

    n = len(den) - 1
    poles = np.roots(den)
    if len(num) >= len(den):
        k_coeffs, num = np.polydiv(num, den)
    else:
        k_coeffs = np.array([])
    residues = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        num_val = np.polyval(num, poles[i])
        den_deriv = np.polyder(den)
        den_val = np.polyval(den_deriv, poles[i])
        if den_val != 0:
            residues[i] = num_val / den_val
    return residues, poles, k_coeffs


trnfn = transfer_function


def cheatsheet() -> str:
    return "transfer_function({}) -> Transfer function step and impulse response."
