# morie.fn — function file (hadesllm/morie)
"""Bayesian ideal-point estimation (Armstrong Ch 5)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_ideal_points", "bysid"]


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bayesian_ideal_points(x, n_iter: int = 400, burn: int = 100,
                          seed: int = 0):
    """Bayesian ideal-point estimation, Metropolis-within-Gibbs surrogate
    for Clinton-Jackman-Rivers (2004).

    Prior: x_i ~ N(0, 1), alpha_j ~ N(0, 5), beta_j ~ N(0, 5).
    Likelihood: P(yea_ij) = sigmoid(alpha_j*(x_i - beta_j)).

    Parameters
    ----------
    x : (n, m) binary vote matrix, or 1-D vector (treated as single item).
    n_iter, burn : MCMC sweep lengths.
    seed : RNG seed.

    Returns
    -------
    RichResult with keys: x_mean, x_sd, x_ci, alpha, beta, n_iter
    """
    rng = np.random.default_rng(seed)
    M = np.asarray(x, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n, m = M.shape
    if n < 2:
        return RichResult(payload={"x_mean": np.full(n, np.nan),
                                   "x_sd": np.full(n, np.nan),
                                   "x_ci": np.full((n, 2), np.nan),
                                   "alpha": np.full(m, np.nan),
                                   "beta": np.full(m, np.nan),
                                   "n_iter": 0,
                                   "method": "bayesian_ideal_points"})
    # Init from SVD
    Mc = np.nan_to_num(M - np.nanmean(M, axis=0, keepdims=True))
    try:
        u, s, vt = np.linalg.svd(Mc, full_matrices=False)
        x_cur = u[:, 0] * s[0]
    except np.linalg.LinAlgError:
        x_cur = rng.normal(size=n)
    x_cur = (x_cur - x_cur.mean()) / (x_cur.std() + 1e-12)
    a_cur = np.ones(m)
    b_cur = np.zeros(m)
    samples = []
    a_samples = []
    b_samples = []
    step_x = 0.4
    step_ab = 0.3

    def loglik(xv, av, bv):
        Z = av[None, :] * (xv[:, None] - bv[None, :])
        P = _logistic(Z)
        mask = ~np.isnan(M)
        ll = np.where(mask,
                      M * np.log(P + 1e-12) + (1 - M) * np.log(1 - P + 1e-12),
                      0.0)
        return float(np.sum(ll))

    ll_cur = loglik(x_cur, a_cur, b_cur)
    for t in range(n_iter):
        # Metropolis on x
        x_prop = x_cur + step_x * rng.normal(size=n)
        ll_prop = loglik(x_prop, a_cur, b_cur)
        log_a = (ll_prop - 0.5 * np.sum(x_prop ** 2)) \
              - (ll_cur - 0.5 * np.sum(x_cur ** 2))
        if np.log(rng.uniform()) < log_a:
            x_cur, ll_cur = x_prop, ll_prop
        # Metropolis on alpha
        a_prop = a_cur + step_ab * rng.normal(size=m)
        ll_prop = loglik(x_cur, a_prop, b_cur)
        log_a = (ll_prop - 0.5 * np.sum(a_prop ** 2) / 25.0) \
              - (ll_cur - 0.5 * np.sum(a_cur ** 2) / 25.0)
        if np.log(rng.uniform()) < log_a:
            a_cur, ll_cur = a_prop, ll_prop
        # Metropolis on beta
        b_prop = b_cur + step_ab * rng.normal(size=m)
        ll_prop = loglik(x_cur, a_cur, b_prop)
        log_a = (ll_prop - 0.5 * np.sum(b_prop ** 2) / 25.0) \
              - (ll_cur - 0.5 * np.sum(b_cur ** 2) / 25.0)
        if np.log(rng.uniform()) < log_a:
            b_cur, ll_cur = b_prop, ll_prop
        if t >= burn:
            # Renorm for identification
            xs = (x_cur - x_cur.mean()) / (x_cur.std() + 1e-12)
            samples.append(xs.copy())
            a_samples.append(a_cur.copy())
            b_samples.append(b_cur.copy())
    arr = np.array(samples) if samples else np.zeros((1, n))
    x_mean = arr.mean(axis=0)
    x_sd = arr.std(axis=0)
    x_ci = np.percentile(arr, [2.5, 97.5], axis=0).T
    a_mean = np.mean(a_samples, axis=0) if a_samples else np.full(m, np.nan)
    b_mean = np.mean(b_samples, axis=0) if b_samples else np.full(m, np.nan)
    return RichResult(
        title="Bayesian ideal points (Metropolis-within-Gibbs)",
        summary_lines=[("posterior draws", len(samples)),
                       ("n legislators", n), ("m items", m)],
        payload={"x_mean": x_mean, "x_sd": x_sd, "x_ci": x_ci,
                 "alpha": a_mean, "beta": b_mean,
                 "n_iter": int(n_iter),
                 "method": "bayesian_ideal_points"},
    )


bysid = bayesian_ideal_points


def cheatsheet():
    return "bysid: Bayesian ideal points (CJR Metropolis-within-Gibbs)."


# CANONICAL TEST
# >>> rng = np.random.default_rng(1)
# >>> X = rng.normal(size=15)
# >>> Y = (X[:, None] + rng.normal(size=(15, 8)) > 0).astype(int)
# >>> r = bayesian_ideal_points(Y, n_iter=200, burn=50)
# >>> assert abs(np.corrcoef(r["x_mean"], X)[0,1]) > 0.4
