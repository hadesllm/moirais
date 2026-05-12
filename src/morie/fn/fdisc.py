# morie.fn -- function file (hadesllm/morie)
"""f-divergence (general: KL, chi-squared, Hellinger, total variation)."""

__all__ = ["fdisc"]

import numpy as np
from ._richresult import RichResult


def fdisc(
    p: np.ndarray,
    q: np.ndarray,
    *,
    divergence: str = "kl",
) -> dict:
    r"""
    Compute an f-divergence between two discrete distributions.

    .. math::

        D_f(P \\| Q) = \\sum_x q(x) \\, f\\!\\left(\\frac{p(x)}{q(x)}\\right)

    Parameters
    ----------
    p : np.ndarray
        First distribution, shape (n,). Must sum to 1.
    q : np.ndarray
        Second distribution, shape (n,). Must sum to 1.
    divergence : str
        Type: 'kl' (Kullback-Leibler), 'reverse_kl', 'chi2' (chi-squared),
        'hellinger', 'tv' (total variation), 'js' (Jensen-Shannon).

    Returns
    -------
    dict
        'divergence' (float), 'type' (str), 'symmetric' (bool).

    Raises
    ------
    ValueError
        If distributions invalid or unknown divergence type.

    References
    ----------
    Csiszar, I. (1967). Information-type measures of difference of
    probability distributions. Studia Sci. Math. Hungar., 2, 299-318.
    Ali, S. M. & Silvey, S. D. (1966). A general class of coefficients of
    divergence. J. R. Stat. Soc. B, 28, 131-142.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    if not np.isclose(p.sum(), 1.0):
        raise ValueError("p must sum to 1.")
    if not np.isclose(q.sum(), 1.0):
        raise ValueError("q must sum to 1.")
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")

    eps = 1e-300

    if divergence == "kl":
        mask = p > eps
        val = float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps))))
        sym = False
    elif divergence == "reverse_kl":
        mask = q > eps
        val = float(np.sum(q[mask] * np.log(q[mask] / (p[mask] + eps))))
        sym = False
    elif divergence == "chi2":
        mask = q > eps
        val = float(np.sum((p[mask] - q[mask]) ** 2 / q[mask]))
        sym = False
    elif divergence == "hellinger":
        val = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
        sym = True
    elif divergence == "tv":
        val = float(0.5 * np.sum(np.abs(p - q)))
        sym = True
    elif divergence == "js":
        m = 0.5 * (p + q)
        kl_pm = np.sum(p[p > eps] * np.log(p[p > eps] / (m[p > eps] + eps)))
        kl_qm = np.sum(q[q > eps] * np.log(q[q > eps] / (m[q > eps] + eps)))
        val = float(0.5 * (kl_pm + kl_qm))
        sym = True
    else:
        raise ValueError(
            f"Unknown divergence: {divergence!r}. "
            "Use 'kl', 'reverse_kl', 'chi2', 'hellinger', 'tv', or 'js'."
        )

    return RichResult(payload={"divergence": max(val, 0.0), "type": divergence, "symmetric": sym})
