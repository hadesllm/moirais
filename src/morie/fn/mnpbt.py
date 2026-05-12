# morie.fn — function file (hadesllm/morie)
"""Multinomial probit for spatial choice (Armstrong Ch 9)."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["multinomial_probit_spatial", "mnpbt"]


def multinomial_probit_spatial(x, n_draws: int = 2000, seed: int = 0):
    """Multinomial probit choice probabilities under independent Gumbel
    error replaced by Gaussian (the "probit" surrogate for MNL).

    Computes P(choice j) for utility matrix U of shape (n_obs, n_alt):
        eps ~ N(0, I), choice = argmax_j (U_ij + eps_j)
    Probability estimated by GHK-style Monte Carlo (independent Gaussian
    errors — sufficient for the IIA-relaxed spatial-choice case in
    Armstrong Ch 9).

    Parameters
    ----------
    x : (n_obs, n_alt) deterministic utility matrix U.
        A 1-D vector is interpreted as a single observation's utilities.

    Returns
    -------
    RichResult with keys: probs, max_alt, n_obs, n_alt
    """
    U = np.asarray(x, dtype=float)
    if U.ndim == 1:
        U = U.reshape(1, -1)
    n, J = U.shape
    if J < 2:
        return RichResult(payload={"probs": np.ones((n, J)),
                                   "max_alt": np.zeros(n, dtype=int),
                                   "n_obs": n, "n_alt": J,
                                   "method": "multinomial_probit"})
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal(size=(n_draws, n, J))
    Y = U[None, :, :] + draws
    picks = np.argmax(Y, axis=2)
    probs = np.zeros((n, J))
    for j in range(J):
        probs[:, j] = np.mean(picks == j, axis=0)
    # Closed-form sanity check for the binary case (J = 2) via Phi
    if J == 2:
        probs[:, 1] = norm.cdf((U[:, 1] - U[:, 0]) / np.sqrt(2.0))
        probs[:, 0] = 1.0 - probs[:, 1]
    max_alt = np.argmax(probs, axis=1)
    return RichResult(
        title="Multinomial probit (spatial choice)",
        summary_lines=[("n observations", n), ("n alternatives", J),
                       ("MC draws", int(n_draws))],
        payload={"probs": probs, "max_alt": max_alt,
                 "n_obs": int(n), "n_alt": int(J),
                 "method": "multinomial_probit"},
    )


mnpbt = multinomial_probit_spatial


def cheatsheet():
    return "mnpbt: Multinomial probit — GHK Monte-Carlo choice probs."


# CANONICAL TEST
# >>> r = multinomial_probit_spatial(np.array([[1.0, 0.0]]), n_draws=500)
# >>> assert r["probs"][0, 0] > 0.6
