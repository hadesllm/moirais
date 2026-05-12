# morie.fn -- function file (hadesllm/morie)
"""Hilbert-transform envelope -- Rangayyan Ch 5."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_envelope"]


def rangayyan_envelope(x):
    """Analytic-signal envelope via the Hilbert transform.

    ``env(t) = |x(t) + j H{x(t)}|`` where H{·} is the discrete Hilbert
    transform (``scipy.signal.hilbert``).

    Parameters
    ----------
    x : array-like

    Returns
    -------
    RichResult with keys ``envelope``, ``analytic``,
    ``instantaneous_phase``, ``instantaneous_freq``.

    References
    ----------
    Rangayyan Ch 5.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=float)
    z = hilbert(x)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase) / (2 * np.pi)
    res = RichResult(
        title="Hilbert envelope",
        summary_lines=[
            ("N samples", int(x.size)),
            ("Envelope mean", float(env.mean())),
            ("Envelope max", float(env.max())),
        ],
        interpretation=f"Analytic envelope mean {env.mean():.4g}, peak {env.max():.4g}.",
        payload={"envelope": env, "analytic": z,
                 "instantaneous_phase": phase,
                 "instantaneous_freq": inst_freq},
    )
    return with_describe_pointer(res, "rgenv")


# CANONICAL TEST
# >>> t = np.arange(100)/100.0
# >>> x = np.cos(2*np.pi*5*t) * (1 + 0.3*np.cos(2*np.pi*0.5*t))
# >>> r = rangayyan_envelope(x)
# >>> r["envelope"].shape == x.shape
# True


def cheatsheet():
    return "rgenv: Hilbert envelope -- Rangayyan Ch 5"
