# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian reliable change index."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp
from ._richresult import RichResult


def bayesian_rci(pre: np.ndarray, post: np.ndarray, cdf=None, *, n_iter: int = 2000, sem: float | None = None, seed: int = 42) -> dict:
    """Bayesian Reliable Change Index.

    Computes the posterior probability that each individual's change
    exceeds measurement error.  Traditional RCI (Jacobson & Truax, 1991)
    is a special case.

    Parameters
    ----------
    pre : array-like
        Pre-intervention scores.
    post : array-like
        Post-intervention scores.
    n_iter : int
        MCMC iterations (default 2000).
    sem : float, optional
        Standard Error of Measurement.  If *None*, estimated from
        the data using test-retest correlation.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``rci_classical`` (array), ``prob_reliable_change``
        (array), ``n_improved``, ``n_deteriorated``, ``n_unchanged``,
        ``sem``, ``n``.

    References
    ----------
    Jacobson, N. S., & Truax, P. (1991). Clinical significance: a
    statistical approach to defining meaningful change. *Journal of
    Consulting and Clinical Psychology*, 59(1), 12--19.
    """
    pre_a = np.asarray(pre, dtype=np.float64).ravel()
    post_a = np.asarray(post, dtype=np.float64).ravel()
    mask = np.isfinite(pre_a) & np.isfinite(post_a)
    pre_a, post_a = pre_a[mask], post_a[mask]
    n = len(pre_a)
    rng = np.random.default_rng(seed)

    if n < 3:
        return RichResult(payload={"rci_classical": np.array([]), "prob_reliable_change": np.array([]), "n": n})

    # Estimate SEM if not provided
    if sem is None:
        r_tt = np.corrcoef(pre_a, post_a)[0, 1]
        sd_pre = np.std(pre_a, ddof=1)
        sem = sd_pre * np.sqrt(1 - r_tt)

    s_diff = np.sqrt(2 * sem**2)

    # Classical RCI
    change = post_a - pre_a
    rci = change / s_diff

    # Bayesian: posterior probability of reliable change
    # Model: true_change_i ~ N(mu_change, sigma_change^2)
    # Observed change_i ~ N(true_change_i, s_diff^2)
    # Prior: mu_change ~ N(0, 10), sigma_change ~ HalfCauchy(5)

    burn = n_iter // 2
    mu_samples = np.zeros(n_iter)
    sigma_samples = np.zeros(n_iter)
    mu = np.mean(change)
    sigma = np.std(change, ddof=1)

    for t in range(n_iter):
        # Update mu
        prec = n / (sigma**2 + s_diff**2) + 1 / 100
        post_mean = (np.sum(change) / (sigma**2 + s_diff**2)) / prec
        mu = rng.normal(post_mean, 1.0 / np.sqrt(prec))

        # Update sigma (MH step)
        sigma_prop = sigma + rng.standard_normal() * 0.5
        if sigma_prop > 0:
            ll_curr = np.sum(sp.norm.logpdf(change, mu, np.sqrt(sigma**2 + s_diff**2)))
            ll_prop = np.sum(sp.norm.logpdf(change, mu, np.sqrt(sigma_prop**2 + s_diff**2)))
            # Half-Cauchy prior
            ll_curr += sp.halfcauchy.logpdf(sigma, scale=5)
            ll_prop += sp.halfcauchy.logpdf(sigma_prop, scale=5)
            if np.log(rng.random()) < ll_prop - ll_curr:
                sigma = sigma_prop

        mu_samples[t] = mu
        sigma_samples[t] = sigma

    # Posterior probability of reliable change for each person
    mu_post = mu_samples[burn:]
    sigma_post = sigma_samples[burn:]
    prob_reliable = np.zeros(n)

    for i in range(n):
        # Posterior of true change for person i
        # Shrinkage: true_i ~ N(weighted avg of change_i and mu, ...)
        total_var = sigma_post**2 + s_diff**2
        weight = sigma_post**2 / total_var
        post_true_mean = weight * change[i] + (1 - weight) * mu_post
        post_true_var = sigma_post**2 * s_diff**2 / total_var
        # P(|true_change| > 1.96 * SEM)
        threshold = 1.96 * sem
        p_above = 1 - sp.norm.cdf(threshold, post_true_mean, np.sqrt(post_true_var))
        p_below = sp.norm.cdf(-threshold, post_true_mean, np.sqrt(post_true_var))
        prob_reliable[i] = float(np.mean(p_above + p_below))

    n_improved = int(np.sum(rci > 1.96))
    n_deteriorated = int(np.sum(rci < -1.96))
    n_unchanged = n - n_improved - n_deteriorated

    return {
        "rci_classical": rci,
        "prob_reliable_change": prob_reliable,
        "n_improved": n_improved,
        "n_deteriorated": n_deteriorated,
        "n_unchanged": n_unchanged,
        "sem": float(sem),
        "n": n,
    }


def cheatsheet() -> str:
    return "bayesian_rci({}) -> Bayesian reliable change index."
