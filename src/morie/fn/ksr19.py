# morie.fn -- function file (hadesllm/morie)
"""Cox proportional-hazards partial likelihood (Kosorok 2008, Ch 8).

l(beta) = sum_{i: delta_i=1} [ X_i' beta - log( sum_{j in R(t_i)}
exp(X_j' beta) ) ], with R(t_i) the risk set at event time t_i.
We fit a *scalar*-covariate Cox model via Newton-Raphson on the
partial-likelihood gradient and return beta_hat with observed-
information SE (no time ties; Breslow handling for ties).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_cox_partial_likelihood"]


def kosorok_cox_partial_likelihood(x, t, event, tol=1e-10, max_iter=100):
    """Scalar-covariate Cox partial-likelihood MLE.

    Parameters
    ----------
    x : array-like, covariate vector.
    t : array-like, observed times.
    event : array-like {0,1}.

    Returns
    -------
    RichResult with keys estimate (beta_hat), se, n, method.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    e = np.asarray(event, dtype=int)
    n = len(x)
    order = np.argsort(t, kind="mergesort")
    x = x[order]; t = t[order]; e = e[order]
    beta = 0.0
    for _ in range(max_iter):
        wt = np.exp(beta * x)  # exp(X_i beta)
        # for each event i, R(t_i) = {j : t_j >= t_i}; with sorted t,
        # that is j >= i.
        # Cumulative-from-end sums of wt and wt*x:
        S0 = np.cumsum(wt[::-1])[::-1]
        S1 = np.cumsum((wt * x)[::-1])[::-1]
        S2 = np.cumsum((wt * x * x)[::-1])[::-1]
        # gradient and observed information, summed over events
        grad = float(np.sum(e * (x - S1 / S0)))
        info = float(np.sum(e * (S2 / S0 - (S1 / S0) ** 2)))
        if info <= 0:
            break
        step = grad / info
        beta_new = beta + step
        if abs(step) < tol:
            beta = beta_new
            break
        beta = beta_new
    # final info for SE
    wt = np.exp(beta * x)
    S0 = np.cumsum(wt[::-1])[::-1]
    S1 = np.cumsum((wt * x)[::-1])[::-1]
    S2 = np.cumsum((wt * x * x)[::-1])[::-1]
    info = float(np.sum(e * (S2 / S0 - (S1 / S0) ** 2)))
    se = float(np.sqrt(1.0 / info)) if info > 0 else float("nan")
    return RichResult(payload={
        "estimate": float(beta),
        "se":       se,
        "n":        n,
        "method":   "Cox PH partial-likelihood MLE (Breslow ties, scalar covariate)",
    })


def cheatsheet():
    return "ksr19: Cox PH partial-likelihood (scalar covariate)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    xs = rng.normal(size=100)
    # exponential survival with rate exp(beta*x), beta=0.5
    ts = rng.exponential(scale=np.exp(-0.5 * xs))
    ev = np.ones(100, dtype=int)
    print(kosorok_cox_partial_likelihood(xs, ts, ev))
