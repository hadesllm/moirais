# moirais.fn — function file (hadesllm/moirais)
"""Signal envelope via Hilbert transform. 'We are what they grow beyond.'"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from ._containers import DescriptiveResult


def envelope(
    signal: np.ndarray,
    method: str = "hilbert",
) -> DescriptiveResult:
    """Compute the amplitude envelope of a signal.

    The analytic signal is obtained via the Hilbert transform, and the
    envelope is its absolute value (instantaneous amplitude).

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input signal.
    method : str
        Envelope method. Currently 'hilbert' (analytic signal).

    Returns
    -------
    DescriptiveResult
        name='Signal Envelope', value=mean envelope amplitude,
        extra has 'envelope' (ndarray), 'instantaneous_phase' (ndarray),
        'analytic_signal' (ndarray), 'method'.

    References
    ----------
    Gabor, D. (1946). Theory of communication. Part 1: The analysis
    of information. *Journal of the IEE*, 93(26), 429-441.
    doi:10.1049/ji-3-2.1946.0074
    """
    x = np.asarray(signal, dtype=np.float64).ravel()

    if method != "hilbert":
        raise ValueError(f"Unknown method '{method}'; supported: 'hilbert'")

    analytic = hilbert(x)
    env = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))

    return DescriptiveResult(
        name="Signal Envelope",
        value=float(np.mean(env)),
        extra={
            "envelope": env,
            "instantaneous_phase": inst_phase,
            "analytic_signal": analytic,
            "method": method,
            "n_samples": len(x),
        },
    )


def cheatsheet() -> str:
    return "envelope({}) -> Signal envelope via Hilbert transform. 'We are what they gro"
