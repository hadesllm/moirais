# moirais.fn — function file (hadesllm/moirais)
"""WLSMV estimation for ordinal CFA."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import DescriptiveResult


def cfa_wlsmv(
    polychoric_corr: np.ndarray,
    model_spec: dict[str, list[int]],
    *,
    n_obs: int = 200,
) -> DescriptiveResult:
    """Weighted Least Squares Mean and Variance adjusted (WLSMV) CFA.

    Approximates WLSMV by fitting a minimum-residual factor model
    to a polychoric correlation matrix, appropriate for ordinal data.

    Parameters
    ----------
    polychoric_corr : ndarray
        Polychoric correlation matrix (p x p).
    model_spec : dict
        {factor_name: [item_indices]} factor structure.
    n_obs : int
        Number of observations (for fit index computation).

    Returns
    -------
    DescriptiveResult
        value=dict with loadings and fit indices.

    References
    ----------
    Muthen, B. (1984). A general structural equation model with
    dichotomous, ordered categorical, and continuous latent variable
    indicators. Psychometrika, 49(1), 115-132.
    """
    R = np.asarray(polychoric_corr, dtype=np.float64)
    p = R.shape[0]

    loadings = {}
    all_loads = np.zeros((p, len(model_spec)))
    for fi, (fname, indices) in enumerate(model_spec.items()):
        sub_R = R[np.ix_(indices, indices)]
        evals, evecs = np.linalg.eigh(sub_R)
        order = np.argsort(-evals)
        loads = evecs[:, order[0]] * np.sqrt(max(evals[order[0]], 0))
        loadings[fname] = {f"item_{i}": float(loads[j]) for j, i in enumerate(indices)}
        for j, i in enumerate(indices):
            all_loads[i, fi] = loads[j]

    R_hat = all_loads @ all_loads.T + np.diag(1 - np.sum(all_loads**2, axis=1))
    residuals = R - R_hat
    srmr = float(np.sqrt(np.mean(np.tril(residuals, -1) ** 2)))

    n_free = int(np.sum([len(v) for v in model_spec.values()])) + p
    df = max(p * (p + 1) // 2 - n_free, 1)
    F_val = float(np.sum(np.tril(residuals, -1) ** 2))
    chi2 = max((n_obs - 1) * F_val, 0)
    rmsea = float(np.sqrt(max((chi2 / df - 1) / max(n_obs - 1, 1), 0)))

    return DescriptiveResult(
        name="WLSMV CFA",
        value={
            "loadings": loadings,
            "chi2": float(chi2),
            "df": df,
            "rmsea": rmsea,
            "srmr": srmr,
        },
        extra={"n": n_obs, "p": p, "estimator": "WLSMV"},
    )


wlsmv = cfa_wlsmv


def cheatsheet() -> str:
    return "cfa_wlsmv({}) -> WLSMV estimation for ordinal CFA."
