# moirais.fn — function file (hadesllm/moirais)
"""Markov-switching AR model (Hamilton regime switching)."""

import numpy as np

from ._containers import DescriptiveResult


def ms_ar(y: np.ndarray, p: int = 1, n_regimes: int = 2, max_iter: int = 100) -> DescriptiveResult:
    """
    Markov-switching autoregressive model via EM algorithm.

    Estimates regime-dependent AR coefficients and transition
    probabilities using a simplified EM procedure.

    :param y: 1-D time series.
    :param p: AR order. Default 1.
    :param n_regimes: Number of regimes. Default 2.
    :param max_iter: Max EM iterations. Default 100.
    :return: DescriptiveResult with regime parameters and probabilities.
    :raises ValueError: If series too short.

    References
    ----------
    Hamilton J.D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*,
    57(2), 357-384.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < p + 20:
        raise ValueError(f"Need at least {p + 20} observations, got {n}.")
    T = n - p
    dep = y[p:]
    X = np.column_stack([np.ones(T)] + [y[p - i - 1 : n - i - 1] for i in range(p)])
    rng = np.random.default_rng(42)
    mu = np.array([np.mean(dep) + (i - n_regimes / 2) * np.std(dep) for i in range(n_regimes)])
    sigma2 = np.full(n_regimes, float(np.var(dep)))
    betas = np.zeros((n_regimes, p + 1))
    for r in range(n_regimes):
        betas[r, 0] = mu[r]
    trans = np.full((n_regimes, n_regimes), 1.0 / n_regimes)
    for _ in range(max_iter):
        log_lik = np.zeros((T, n_regimes))
        for r in range(n_regimes):
            resid = dep - X @ betas[r]
            s2 = max(sigma2[r], 1e-10)
            log_lik[:, r] = -0.5 * (np.log(2 * np.pi * s2) + resid ** 2 / s2)
        max_ll = np.max(log_lik, axis=1, keepdims=True)
        lik = np.exp(log_lik - max_ll)
        gamma = lik / lik.sum(axis=1, keepdims=True)
        for r in range(n_regimes):
            w = gamma[:, r]
            W = np.diag(w)
            XtWX = X.T @ W @ X
            XtWy = X.T @ (w * dep)
            try:
                betas[r] = np.linalg.solve(XtWX + 1e-8 * np.eye(p + 1), XtWy)
            except np.linalg.LinAlgError:
                pass
            resid = dep - X @ betas[r]
            sigma2[r] = max(float(np.sum(w * resid ** 2) / max(np.sum(w), 1e-10)), 1e-10)
        for r in range(n_regimes):
            for s in range(n_regimes):
                trans[r, s] = max(np.sum(gamma[:-1, r] * gamma[1:, s]), 1e-10)
            trans[r] /= trans[r].sum()
    regime_probs = gamma
    most_likely = np.argmax(gamma, axis=1)
    return DescriptiveResult(
        name="ms_ar",
        value=float(np.sum(np.max(log_lik, axis=1))),
        extra={
            "betas": betas.tolist(),
            "sigma2": sigma2.tolist(),
            "transition_matrix": trans.tolist(),
            "regime_probs": regime_probs,
            "most_likely_regime": most_likely,
            "n_regimes": n_regimes,
            "p": p,
            "n": n,
        },
    )


mstar = ms_ar


def cheatsheet() -> str:
    return "ms_ar({}) -> Markov-switching AR model."
