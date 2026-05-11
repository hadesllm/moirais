# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bode magnitude and phase plot computation."""

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Time discovers truth. — Seneca"


def bode_plot(
    num,
    den,
    freq=None,
    n_points: int = 500,
    **kwargs,
) -> DescriptiveResult:
    """
    Compute Bode magnitude and phase for a transfer function.

    Evaluates :math:`H(j\\omega) = N(j\\omega) / D(j\\omega)` over a range
    of frequencies to produce magnitude (dB) and phase (degrees).

    .. math::

        |H(j\\omega)|_{\\text{dB}} = 20 \\log_{10} |H(j\\omega)|

    :param num: Numerator polynomial coefficients (highest power first).
    :param den: Denominator polynomial coefficients (highest power first).
    :param freq: Frequency vector in rad/s. If None, auto-generated logspace.
    :param n_points: Number of frequency points if freq is None. Default 500.
    :return: DescriptiveResult with gain margin and phase margin.
    :raises ValueError: If den is all zeros.

    References
    ----------
    Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall.
    Dorf, R. C. & Bishop, R. H. (2017). *Modern Control Systems* (13th ed.).
    """
    num = np.atleast_1d(np.asarray(num, dtype=np.float64))
    den = np.atleast_1d(np.asarray(den, dtype=np.float64))

    if np.all(den == 0):
        raise ValueError("Denominator coefficients cannot all be zero.")

    if freq is None:
        freq = np.logspace(-2, 3, n_points)
    else:
        freq = np.asarray(freq, dtype=np.float64)

    jw = 1j * freq
    h = np.polyval(num, jw) / np.polyval(den, jw)

    magnitude_db = 20.0 * np.log10(np.abs(h) + 1e-300)
    phase_deg = np.degrees(np.unwrap(np.angle(h)))

    gain_crossover_idx = np.where(np.diff(np.sign(magnitude_db)))[0]
    phase_crossover_idx = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]

    gain_margin = float("inf")
    if len(phase_crossover_idx) > 0:
        idx = phase_crossover_idx[0]
        gain_margin = float(-magnitude_db[idx])

    phase_margin = float("inf")
    if len(gain_crossover_idx) > 0:
        idx = gain_crossover_idx[0]
        phase_margin = float(180.0 + phase_deg[idx])

    return DescriptiveResult(
        name="bode_plot",
        value=gain_margin,
        extra={
            "gain_margin_db": gain_margin,
            "phase_margin_deg": phase_margin,
            "frequency": freq,
            "magnitude_db": magnitude_db,
            "phase_deg": phase_deg,
        },
    )


bode = bode_plot


def cheatsheet() -> str:
    return "bode_plot({}) -> Bode magnitude and phase plot computation."
