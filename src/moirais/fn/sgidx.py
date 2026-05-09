"""Index of dispersion (VMR) for point counts."""

from __future__ import annotations

from ._containers import DescriptiveResult


def index_of_dispersion(counts, cdf=None):
    """Compute the variance-to-mean ratio (index of dispersion).

    VMR = 1 indicates CSR, > 1 clustering, < 1 regularity.

    .. epigraph:: "Accio!" -- Harry Potter, Harry Potter

    Parameters
    ----------
    counts : array_like
        Quadrat counts.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy import stats

    c = np.asarray(counts, dtype=np.float64).ravel()
    mean_c = float(c.mean())
    var_c = float(c.var(ddof=1)) if len(c) > 1 else 0.0

    vmr = var_c / mean_c if mean_c > 0 else 0.0
    n = len(c)
    chi2 = vmr * (n - 1)
    p_value = 1.0 - stats.chi2.cdf(chi2, n - 1)

    if vmr > 1:
        pattern = "clustered"
    elif vmr < 1:
        pattern = "regular"
    else:
        pattern = "random"

    return DescriptiveResult(
        name="index_of_dispersion",
        value=vmr,
        extra={
            "VMR": vmr,
            "chi2": chi2,
            "p_value": float(p_value),
            "df": n - 1,
            "mean_count": mean_c,
            "var_count": var_c,
            "pattern": pattern,
        },
    )


sgidx = index_of_dispersion


def cheatsheet() -> str:
    return "index_of_dispersion({}) -> Index of dispersion (VMR) for point counts."
