"""Mantel test for spatial correlation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def mantel_test(D_spatial, D_attribute, n_perm=999):
    """Mantel test comparing spatial and attribute distance matrices.

    .. epigraph:: It is not what happens to you, but how you react, that matters. -- Epictetus

    Parameters
    ----------
    D_spatial : array_like
        Spatial distance matrix, shape ``(n, n)``.
    D_attribute : array_like
        Attribute distance matrix, shape ``(n, n)``.
    n_perm : int
        Number of permutations.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Ds = np.asarray(D_spatial, dtype=np.float64)
    Da = np.asarray(D_attribute, dtype=np.float64)
    n = Ds.shape[0]

    idx = np.triu_indices(n, k=1)
    ds = Ds[idx]
    da = Da[idx]

    def _corr(x, y):
        xc = x - x.mean()
        yc = y - y.mean()
        denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
        return np.sum(xc * yc) / denom if denom > 0 else 0.0

    observed = _corr(ds, da)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Da_perm = Da[np.ix_(perm, perm)]
        da_perm = Da_perm[idx]
        if _corr(ds, da_perm) >= observed:
            count += 1

    p_value = (count + 1) / (n_perm + 1)

    return DescriptiveResult(
        name="mantel_test",
        value=float(observed),
        extra={
            "statistic": float(observed),
            "p_value": p_value,
            "n_perm": n_perm,
            "significant": p_value < 0.05,
        },
    )


sgmnt = mantel_test


def cheatsheet() -> str:
    return "mantel_test({}) -> Mantel test for spatial correlation."
