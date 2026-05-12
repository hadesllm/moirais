# morie.fn -- function file (hadesllm/morie)
"""M-estimator with a nuisance parameter (Kosorok 2008, Ch 5).

theta_n = argmax_theta P_n m(.; theta, eta_n), with eta_n a
plug-in nuisance estimate.  We implement Huber-M location with
profiled scale: eta_n = MAD(x)/0.6745, then iteratively-reweighted
LS to find theta_n that solves sum psi_H((x - theta)/eta_n) = 0
where psi_H is the Huber score with tuning k = 1.345.  Returns
theta_n and its sandwich SE (Huber 1981, Section 3.2.2).
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_m_estimator"]


def _huber_psi(u, k=1.345):
    return np.clip(u, -k, k)


def kosorok_m_estimator(x, y=None, k=1.345, max_iter=100, tol=1e-10):
    """Huber-M location estimator with profiled scale eta = MAD/0.6745.

    Parameters
    ----------
    x : array-like.
    y : ignored (kept for API parity).
    k : float, Huber tuning constant (default 1.345 = 95% efficiency).

    Returns
    -------
    RichResult with: estimate (theta_n), se (sandwich SE), n, method.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    eta = float(np.median(np.abs(x - np.median(x))) / 0.6745)
    if eta == 0:
        eta = float(np.std(x, ddof=1)) or 1.0
    theta = float(np.median(x))
    for _ in range(max_iter):
        u = (x - theta) / eta
        psi = _huber_psi(u, k)
        # Newton step: theta += eta * mean(psi) / mean(psi'),
        # with psi' = 1{|u| <= k}.
        denom = float(np.mean(np.abs(u) <= k))
        if denom == 0:
            break
        step = eta * float(np.mean(psi)) / denom
        theta_new = theta + step
        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new
    u = (x - theta) / eta
    psi = _huber_psi(u, k)
    A = float(np.mean(np.abs(u) <= k)) / eta  # E[psi'/eta]
    B = float(np.mean(psi ** 2))
    se = float(np.sqrt(B / (A ** 2) / n))
    return RichResult(payload={
        "estimate": float(theta), "se": se, "n": n,
        "method":   "Huber-M location (k=%.3f) with profiled MAD/0.6745 scale" % k,
    })


def cheatsheet():
    return "ksr10: Huber-M location estimator with profiled scale"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(kosorok_m_estimator(rng.normal(size=200)))
