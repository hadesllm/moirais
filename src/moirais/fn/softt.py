"""Proximal soft thresholding (prox of L1 norm)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luminous beings are we, not this crude matter."


def soft_threshold_fn(x, tau: float = 0.1, **kwargs) -> DescriptiveResult:
    """Proximal soft thresholding: prox_{tau * ||.||_1}(x).

    This is the proximal operator of the scaled L1 norm, commonly used
    in proximal gradient methods (ISTA/FISTA).

    Parameters
    ----------
    x : array-like
        Input vector.
    tau : float
        Proximal step size (default 0.1).

    Returns
    -------
    DescriptiveResult
        ``value`` is L1 norm of result; ``extra`` has ``prox``,
        ``sparsity`` (fraction of zeros), ``tau``.
    """
    x = np.asarray(x, dtype=float)
    prox = np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)
    l1 = float(np.sum(np.abs(prox)))
    n_total = max(1, int(np.prod(x.shape)))
    n_zero = int(np.sum(np.abs(prox) < 1e-15))
    sparsity = n_zero / n_total

    return DescriptiveResult(
        name="soft_threshold_fn",
        value=l1,
        extra={"prox": prox, "sparsity": sparsity, "tau": tau},
    )


softt = soft_threshold_fn


def cheatsheet() -> str:
    return "soft_threshold_fn({}) -> Proximal soft thresholding (prox of L1 norm)."
