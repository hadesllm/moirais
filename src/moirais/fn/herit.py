# moirais.fn — function file (hadesllm/moirais)
"""Heritability estimation (GREML)."""

__all__ = ["herit"]

import numpy as np

from ._containers import GenomicsResult


def herit(
    y: np.ndarray,
    G: np.ndarray,
    *,
    X: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> GenomicsResult:
    """Estimate SNP heritability using GREML (AI-REML).

    Fits a linear mixed model y = Xb + g + e where g ~ N(0, G*var_g)
    and estimates var_g / (var_g + var_e) = h^2.

    Uses the Average Information REML algorithm for variance
    component estimation.

    Parameters
    ----------
    y : array, shape (n,)
        Phenotype vector.
    G : array, shape (n, n)
        Genomic relationship matrix.
    X : array, shape (n, p), optional
        Fixed-effect design matrix.  If None, intercept only.
    max_iter : int
        Maximum AI-REML iterations.
    tol : float
        Convergence tolerance for variance components.

    Returns
    -------
    GenomicsResult
        statistic = h^2 (heritability estimate),
        extra has 'var_g', 'var_e', 'se_h2', 'n_iter'.

    References
    ----------
    Yang, J., et al. (2010). Common SNPs explain a large proportion
        of the heritability for human height. Nature Genetics,
        42(7), 565-569.
    Yang, J., et al. (2011). GCTA: a tool for genome-wide complex
        trait analysis. Am. J. Hum. Genet., 88(1), 76-82.
    """
    y = np.asarray(y, dtype=float).ravel()
    G = np.asarray(G, dtype=float)
    n = len(y)

    if G.shape != (n, n):
        raise ValueError(f"G must be ({n},{n}).")

    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

    var_p = float(np.var(y))
    var_g = var_p * 0.5
    var_e = var_p * 0.5

    for iteration in range(max_iter):
        V = G * var_g + np.eye(n) * var_e
        try:
            V_inv = np.linalg.inv(V + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            break

        P = V_inv - V_inv @ X @ np.linalg.solve(X.T @ V_inv @ X, X.T @ V_inv)

        Py = P @ y

        dL_dg = -0.5 * np.trace(P @ G) + 0.5 * Py.T @ G @ Py
        dL_de = -0.5 * np.trace(P) + 0.5 * Py.T @ Py

        PG = P @ G
        AI_gg = 0.5 * Py.T @ G @ PG @ Py
        AI_ee = 0.5 * Py.T @ P @ Py
        AI_ge = 0.5 * Py.T @ G @ P @ Py

        AI = np.array([[AI_gg, AI_ge], [AI_ge, AI_ee]])
        score = np.array([dL_dg, dL_de])

        try:
            delta = np.linalg.solve(AI + np.eye(2) * 1e-8, score)
        except np.linalg.LinAlgError:
            break

        var_g_new = max(var_g + delta[0], 1e-8)
        var_e_new = max(var_e + delta[1], 1e-8)

        if abs(var_g_new - var_g) < tol and abs(var_e_new - var_e) < tol:
            var_g = var_g_new
            var_e = var_e_new
            break

        var_g = var_g_new
        var_e = var_e_new

    h2 = var_g / (var_g + var_e) if (var_g + var_e) > 0 else 0.0

    try:
        AI_inv = np.linalg.inv(AI + np.eye(2) * 1e-8)
        dh2_dg = var_e / (var_g + var_e) ** 2
        dh2_de = -var_g / (var_g + var_e) ** 2
        grad = np.array([dh2_dg, dh2_de])
        se_h2 = float(np.sqrt(max(grad @ AI_inv @ grad, 0)))
    except (np.linalg.LinAlgError, ValueError):
        se_h2 = float("nan")

    return GenomicsResult(
        name="GREML_h2",
        statistic=float(h2),
        n=n,
        extra={
            "var_g": float(var_g),
            "var_e": float(var_e),
            "se_h2": se_h2,
            "n_iter": iteration + 1,
        },
    )


def cheatsheet() -> str:
    return "herit(y, G) -> SNP heritability via GREML."
