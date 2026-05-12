# morie.fn -- function file (hadesllm/morie)
"""Cutting plane / cutting sphere for vote classification (Armstrong Ch 3)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["cutting_plane_sphere", "csphr"]


def cutting_plane_sphere(x, votes=None):
    """Compute the cutting hyperplane w'x = c that maximises classification
    of binary votes given legislator ideal points.

    Strategy: standardised normal-vector linear discriminant --
        w  ∝  Σ_pooled⁻¹ (μ_yea - μ_nay)
        c  =  w' (μ_yea + μ_nay) / 2
    For p = 1 this collapses to the midpoint between class means.

    Parameters
    ----------
    x : (n, p) ideal-point matrix.  1-D vectors are treated as p = 1.
    votes : (n,) {0,1} -- required for non-trivial output.

    Returns
    -------
    RichResult with keys: w, c, midpoint, correct_class, n, p
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if votes is None or n == 0:
        return RichResult(payload={"w": np.zeros(p), "c": np.nan,
                                   "midpoint": np.full(p, np.nan),
                                   "correct_class": 0, "n": n, "p": p,
                                   "method": "cutting_plane_sphere"})
    y = np.asarray(votes, dtype=int).ravel()
    X_y = X[y == 1]; X_n = X[y == 0]
    if X_y.size == 0 or X_n.size == 0:
        return RichResult(payload={"w": np.zeros(p), "c": np.nan,
                                   "midpoint": np.full(p, np.nan),
                                   "correct_class": int(np.sum(y == y[0])),
                                   "n": n, "p": p,
                                   "method": "cutting_plane_sphere"})
    mu_y = X_y.mean(axis=0); mu_n = X_n.mean(axis=0)
    # Pooled covariance
    S_y = np.cov(X_y, rowvar=False) if X_y.shape[0] > 1 else np.zeros((p, p))
    S_n = np.cov(X_n, rowvar=False) if X_n.shape[0] > 1 else np.zeros((p, p))
    S = ((X_y.shape[0] - 1) * np.atleast_2d(S_y)
         + (X_n.shape[0] - 1) * np.atleast_2d(S_n)) \
        / max(n - 2, 1)
    S = S + 1e-9 * np.eye(p)
    try:
        w = np.linalg.solve(S, mu_y - mu_n)
    except np.linalg.LinAlgError:
        w = mu_y - mu_n
    midpoint = (mu_y + mu_n) / 2
    c = float(w @ midpoint)
    pred = (X @ w > c).astype(int)
    cc = int(np.sum(pred == y))
    # Flip polarity if necessary
    if cc < n - cc:
        w = -w; c = -c; cc = n - cc
    return RichResult(
        title="Cutting plane (linear discriminant)",
        summary_lines=[("Correctly classified", cc), ("c (intercept)", c),
                       ("dim", p), ("n", n)],
        payload={"w": w, "c": c, "midpoint": midpoint,
                 "correct_class": cc, "n": int(n), "p": int(p),
                 "method": "cutting_plane_sphere"},
    )


csphr = cutting_plane_sphere


def cheatsheet():
    return "csphr: Cutting plane w'x=c -- pooled-LDA normal vector."


# CANONICAL TEST
# >>> r = cutting_plane_sphere([-1,-1,1,1], votes=[0,0,1,1])
# >>> assert r["correct_class"] == 4
