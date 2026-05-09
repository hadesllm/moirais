# moirais.fn — function file (hadesllm/moirais)
"""Mahalanobis distance with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def mahalan(x: Union[Sequence, np.ndarray],
            mu: Union[Sequence, np.ndarray],
            cov: Union[Sequence, np.ndarray]):
    """Mahalanobis distance: sqrt((x-mu)' Sigma^-1 (x-mu))."""
    from ._richresult import RichResult
    from scipy.stats import chi2 as _chi2
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if x.shape != mu.shape:
        raise ValueError(f"x and mu shape mismatch: {x.shape} vs {mu.shape}.")
    if cov.shape != (x.size, x.size):
        raise ValueError(f"cov must be {x.size}x{x.size}, got {cov.shape}.")
    diff = x - mu
    warnings = []
    cond = float(np.linalg.cond(cov))
    if cond > 1e10:
        warnings.append(f"covariance is ill-conditioned (cond={cond:.2e}); pinv used.")
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-10):
        warnings.append(f"covariance not PSD (min eig {float(eigvals.min()):.2e}); pinv used.")
    dist = float(np.sqrt(diff @ np.linalg.pinv(cov) @ diff))
    cutoff = float(np.sqrt(_chi2.ppf(0.95, x.size)))
    return RichResult(
        title="Mahalanobis distance",
        summary_lines=[("Distance", dist), ("Squared distance", dist**2),
                       ("Dimension", x.size), ("Cov condition number", cond)],
        warnings=warnings,
        interpretation=(f"chi^2(p) cutoff at alpha=0.05 is {cutoff:.3f}; "
                        f"this point {'IS' if dist > cutoff else 'is not'} "
                        f"an outlier under multivariate Normality."),
        payload={"distance": dist, "squared": dist**2, "dimension": x.size,
                 "cov_condition": cond},
    )
