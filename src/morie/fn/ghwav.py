# morie.fn -- function file (hadesllm/morie)
"""Wavelet spike-and-slab prior for function estimation."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_wavelet_prior"]


def _haar_dwt(y):
    """Single-level discrete Haar wavelet transform (in-place style)."""
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    # Pad to next power of 2.
    L = 1
    while L < n:
        L *= 2
    if L > n:
        y = np.concatenate([y, np.zeros(L - n)])
    coeffs = []
    cur = y.copy()
    while len(cur) > 1:
        a = (cur[0::2] + cur[1::2]) / np.sqrt(2.0)
        d = (cur[0::2] - cur[1::2]) / np.sqrt(2.0)
        coeffs.append(d)
        cur = a
    coeffs.append(cur)         # scaling coefficient(s) at the coarsest level
    return coeffs, L


def _haar_idwt(coeffs, L):
    """Inverse single-level Haar DWT given a list of detail coefficients
    from coarsest (last) to finest (first) and the scaling coefficient
    (last entry of `coeffs`)."""
    cur = coeffs[-1]
    details = list(coeffs[:-1])[::-1]   # finest first -> reverse to coarsest first
    for d in details[::-1]:
        # Wait -- coeffs[:-1] is [d_level1 (finest), d_level2, ...]
        # We need to inverse from coarsest to finest, so iterate in reverse.
        out = np.empty(len(cur) + len(d))
        out[0::2] = (cur + d) / np.sqrt(2.0)
        out[1::2] = (cur - d) / np.sqrt(2.0)
        cur = out
    return cur[:L]


def ghosal_wavelet_prior(x, pi=0.5, sigma=None, noise=None):
    """Posterior mean of a Haar-wavelet spike-and-slab estimator.

    Prior on each wavelet coefficient::
        theta_{jk} ~ pi N(0, sigma_j^2) + (1 - pi) delta_0.

    With independent Gaussian noise of sd ``noise`` the posterior of
    ``theta`` given the noisy wavelet coefficient ``y`` is a
    two-component mixture; the posterior mean is

        E[theta | y] = w(y) * (sigma^2/(sigma^2 + noise^2)) y,

    where ``w(y)`` is the posterior inclusion probability

        w(y) = pi phi(y; 0, sigma^2 + noise^2) /
                ( pi phi(y; 0, sigma^2 + noise^2)
                  + (1 - pi) phi(y; 0, noise^2) ).

    Applied coefficient-wise to the Haar-DWT of ``x``; an inverse DWT
    recovers a denoised signal.  This is the canonical
    Abramovich–Sapatinas–Silverman (1998) BayesThresh estimator and is
    rate-minimax over Besov balls (Ghosal Ch 13).

    Parameters
    ----------
    x : array-like -- observed signal (1D).
    pi : prior inclusion probability.
    sigma : slab sd (if None, set to robust sd of the wavelet
        coefficients).
    noise : observation sd (if None, MAD of finest-level coeffs / 0.6745).

    Returns
    -------
    RichResult with ``estimate`` (mean of denoised signal),
    ``fitted`` (denoised signal), ``noise``, ``sigma``, ``inclusion``
    (mean posterior inclusion).

    References
    ----------
    Abramovich, Sapatinas, Silverman (1998). JRSS-B 60.
    Donoho & Johnstone (1994). Biometrika 81.
    Ghosal & van der Vaart (2017) Ch 13.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 4:
        return RichResult(payload={
            "estimate": float(np.mean(x)) if n else float("nan"),
            "fitted": x.tolist(),
            "n": n,
            "method": "Wavelet prior (n<4)",
        })
    coeffs, L = _haar_dwt(x)
    # Detail coefficients only (not the scaling); first list entry is finest.
    finest = coeffs[0]
    if noise is None:
        # Donoho-Johnstone universal-threshold MAD
        noise = float(np.median(np.abs(finest - np.median(finest))) / 0.6745)
        noise = max(noise, 1e-6)
    if sigma is None:
        all_d = np.concatenate([c.ravel() for c in coeffs[:-1]])
        sigma = float(np.sqrt(max(np.var(all_d) - noise ** 2, 1e-6)))
    sigma = max(float(sigma), 1e-6)
    incl_list = []
    new_coeffs = []
    for d in coeffs[:-1]:
        var_slab = sigma ** 2 + noise ** 2
        log_slab = norm.logpdf(d, loc=0, scale=np.sqrt(var_slab))
        log_spike = norm.logpdf(d, loc=0, scale=noise)
        a = np.log(pi) + log_slab
        b = np.log(1 - pi) + log_spike
        m = np.maximum(a, b)
        w = np.exp(a - m) / (np.exp(a - m) + np.exp(b - m))
        shrink = sigma ** 2 / var_slab
        new_d = w * shrink * d
        incl_list.append(w)
        new_coeffs.append(new_d)
    new_coeffs.append(coeffs[-1])
    # Reconstruct
    cur = new_coeffs[-1]
    for d in new_coeffs[-2::-1]:    # coarsest detail -> finest
        out = np.empty(len(cur) + len(d))
        out[0::2] = (cur + d) / np.sqrt(2.0)
        out[1::2] = (cur - d) / np.sqrt(2.0)
        cur = out
    fitted = cur[:n]
    incl_mean = float(np.mean(np.concatenate(incl_list)))
    return RichResult(payload={
        "estimate": float(np.mean(fitted)),
        "fitted": fitted.tolist(),
        "noise": float(noise),
        "sigma": float(sigma),
        "inclusion": incl_mean,
        "n": n,
        "method": "Haar-wavelet spike-and-slab BayesThresh",
    })


def cheatsheet():
    return "ghwav: Wavelet prior (spike-and-slab)"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghwav import ghosal_wavelet_prior
# >>> rng = np.random.default_rng(0)
# >>> x = np.sin(np.linspace(0, 2*np.pi, 128)) + 0.1*rng.normal(size=128)
# >>> r = ghosal_wavelet_prior(x)
# >>> 0 <= r["inclusion"] <= 1
# True
