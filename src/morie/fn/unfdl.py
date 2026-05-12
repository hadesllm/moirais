# morie.fn — function file (hadesllm/morie)
"""Unfolding analysis for preference data (Coombs 1964; Armstrong Ch 7)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["unfolding_analysis", "unfdl"]


def unfolding_analysis(x, k: int = 2, n_iter: int = 100, tol: float = 1e-6):
    """Metric unfolding (Schönemann 1970) — recover (X, Y) such that
    ||x_i - y_j||  is close to the input preference dissimilarity p_{ij}.

    Closed-form Schönemann solution treating the preference matrix as
    squared distances; identification by centring X+Y.

    Parameters
    ----------
    x : (n_resp, n_stim) preference dissimilarity matrix (Δ).
        Higher = less preferred.
    k : output dimensionality.

    Returns
    -------
    RichResult with keys: X, Y, stress, k, n_resp, n_stim
    """
    P = np.asarray(x, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2 or P.shape[1] < 2:
        return RichResult(payload={"X": np.zeros((0, k)),
                                   "Y": np.zeros((0, k)),
                                   "stress": np.nan, "k": k,
                                   "n_resp": 0, "n_stim": 0,
                                   "method": "unfolding"})
    n, m = P.shape
    # Schönemann-style double centring of P^2 then SVD
    P2 = P ** 2
    rmeans = P2.mean(axis=1, keepdims=True)
    cmeans = P2.mean(axis=0, keepdims=True)
    gmean = P2.mean()
    B = -0.5 * (P2 - rmeans - cmeans + gmean)
    u, s, vt = np.linalg.svd(B, full_matrices=False)
    k_eff = min(k, len(s))
    X = u[:, :k_eff] * np.sqrt(s[:k_eff])
    Y = vt[:k_eff, :].T * np.sqrt(s[:k_eff])
    # Iterative SMACOF-lite refinement
    for _ in range(n_iter):
        diff = X[:, None, :] - Y[None, :, :]
        Dh = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
        ratio = P / Dh
        # Update X
        X_new = (ratio[:, :, None] * (X[:, None, :] - Y[None, :, :])
                 ).sum(axis=1) / m + Y.mean(axis=0)
        Y_new = (ratio[:, :, None] * (Y[None, :, :] - X[:, None, :])
                 ).sum(axis=0) / n + X_new.mean(axis=0)
        delta = max(np.max(np.abs(X_new - X)), np.max(np.abs(Y_new - Y)))
        X, Y = X_new, Y_new
        if delta < tol:
            break
    Dh = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1))
    denom = float(np.sum(P ** 2))
    stress = float(np.sqrt(np.sum((P - Dh) ** 2) / denom)) if denom > 0 \
        else np.nan
    return RichResult(
        title="Metric unfolding (Schönemann)",
        summary_lines=[("Stress", stress), ("k", k_eff),
                       ("n respondents", n), ("n stimuli", m)],
        payload={"X": X, "Y": Y, "stress": stress, "k": int(k_eff),
                 "n_resp": int(n), "n_stim": int(m),
                 "method": "unfolding"},
    )


unfdl = unfolding_analysis


def cheatsheet():
    return "unfdl: Metric unfolding — Schönemann + SMACOF refinement."


# CANONICAL TEST
# >>> P = np.array([[1.,2.,3.],[2.,1.,2.],[3.,2.,1.]])
# >>> r = unfolding_analysis(P, k=1)
# >>> assert r["X"].shape == (3, 1) and r["Y"].shape == (3, 1)
