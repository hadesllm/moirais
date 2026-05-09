# moirais.fn — function file (hadesllm/moirais)
"""Determine optimal number of factors (parallel analysis, MAP, BIC)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def efa_nfactors(
    data: pd.DataFrame | np.ndarray,
    *,
    method: str = "parallel",
    nsim: int = 100,
    seed: int = 42,
) -> dict:
    """Determine the optimal number of factors to extract.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    method : str
        Method: 'parallel' (Horn, 1965), 'map' (Velicer, 1976),
        or 'bic' (minimum BIC). Default 'parallel'.
    nsim : int
        Simulations for parallel analysis (default 100).
    seed : int
        Random seed (default 42).

    Returns
    -------
    dict
        Keys: n_factors, method, eigenvalues (observed), threshold
        (parallel only).

    References
    ----------
    Horn, J.L. (1965). A rationale and test for the number of factors
        in factor analysis. Psychometrika, 30(2), 179-185.
    Velicer, W.F. (1976). Determining the number of components from the
        matrix of partial correlations. Psychometrika, 41(3), 321-327.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(data, dtype=np.float64)
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, k = X.shape

    R = np.corrcoef(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(R))[::-1]

    if method == "parallel":
        sim_eigs = np.zeros((nsim, k))
        for i in range(nsim):
            Rs = np.corrcoef(rng.standard_normal((n, k)), rowvar=False)
            sim_eigs[i] = np.sort(np.linalg.eigvalsh(Rs))[::-1]
        threshold = np.percentile(sim_eigs, 95, axis=0)
        n_factors = max(int(np.sum(eigenvalues > threshold)), 1)
        return {
            "n_factors": n_factors,
            "method": "parallel",
            "eigenvalues": eigenvalues.tolist(),
            "threshold": threshold.tolist(),
        }

    elif method == "map":
        # Velicer's MAP: minimize average squared partial correlation
        map_values = np.zeros(k - 1)
        for m in range(k - 1):
            evals, evecs = np.linalg.eigh(R)
            idx = np.argsort(-evals)
            evecs = evecs[:, idx]
            Pm = evecs[:, : m + 1] @ evecs[:, : m + 1].T
            Rp = R - Pm
            np.fill_diagonal(Rp, 0)
            denom = np.sum(~np.eye(k, dtype=bool))
            map_values[m] = np.sum(Rp**2) / denom
        n_factors = max(int(np.argmin(map_values)) + 1, 1)
        return {
            "n_factors": n_factors,
            "method": "map",
            "eigenvalues": eigenvalues.tolist(),
            "map_values": map_values.tolist(),
        }

    elif method == "bic":
        # BIC for each number of factors
        bic_values = []
        for m in range(1, min(k, n // 2)):
            n_params = k * m - m * (m - 1) // 2 + k
            evals_m = eigenvalues[:m]
            evals_rest = eigenvalues[m:]
            if len(evals_rest) == 0 or np.any(evals_rest <= 0):
                break
            loglik = -n / 2 * (np.sum(np.log(evals_m)) + np.sum(evals_rest))
            bic = -2 * loglik + n_params * np.log(n)
            bic_values.append(bic)
        if not bic_values:
            n_factors = 1
        else:
            n_factors = max(int(np.argmin(bic_values)) + 1, 1)
        return {
            "n_factors": n_factors,
            "method": "bic",
            "eigenvalues": eigenvalues.tolist(),
        }

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'parallel', 'map', or 'bic'.")


def cheatsheet() -> str:
    return "efa_nfactors({}) -> Determine optimal number of factors (parallel analysis, MAP,"
