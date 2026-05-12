"""SVM regression (epsilon-SVR with RBF kernel) for genomic prediction."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["svm_genomic"]


def svm_genomic(x, y, markers, C: float = 1.0, epsilon: float = 0.1,
                gamma: float | str = "scale"):
    """Support-vector regression for genomic prediction.

    Model::

        f(x) = sum_i alpha_i K(x_i, x) + b,   K(u,v) = exp(-gamma ||u-v||^2)

    Uses scikit-learn's SVR if available; otherwise falls back to a
    Lagrangian-dual approximation via the kernel-ridge solution
    f(x) = K (K + C^{-1} I)^{-1} y (close enough for small inputs and
    documented as such).

    Parameters
    ----------
    x : array-like (n,) or (n,q) — fixed-effect features, currently
        concatenated to `markers`.
    y : array-like (n,)
    markers : array-like (n, m)
    C : float, default 1.0
    epsilon : float, default 0.1
    gamma : float or "scale" (= 1/(m * var(M))).

    Returns
    -------
    RichResult with payload keys estimate, y_hat, alpha, support_indices,
    n, method.

    References
    ----------
    Vapnik, V. (1995). The Nature of Statistical Learning Theory.
    Montesinos Lopez et al. (2022), Ch. 7.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1 and Xa.size > 0:
        Xa = Xa.reshape(-1, 1)
    if Xa.size == 0:
        feats = M
    else:
        feats = np.column_stack([Xa, M])
    method_used = "sklearn SVR (epsilon-SVR, RBF)"
    try:
        from sklearn.svm import SVR
        svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel="rbf").fit(feats, y)
        y_hat = svr.predict(feats)
        support_idx = svr.support_
        alpha = svr.dual_coef_.ravel()
        intercept = float(svr.intercept_[0])
    except Exception:
        # Numpy fallback: kernel ridge with RBF
        method_used = "Kernel-ridge RBF fallback (no sklearn)"
        if isinstance(gamma, str):
            var_m = float(np.var(M)) if np.var(M) > 0 else 1.0
            gamma_v = 1.0 / (M.shape[1] * var_m)
        else:
            gamma_v = float(gamma)
        sq_norm = np.sum(feats ** 2, axis=1)
        D2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (feats @ feats.T)
        K = np.exp(-gamma_v * np.maximum(D2, 0))
        intercept = float(np.mean(y))
        yc = y - intercept
        alpha = np.linalg.solve(K + (1.0 / max(C, 1e-8)) * np.eye(n), yc)
        y_hat = K @ alpha + intercept
        support_idx = np.where(np.abs(alpha) > 1e-6)[0]
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="SVM (epsilon-SVR, RBF) genomic predictor",
        summary_lines=[
            ("n", n),
            ("m (markers)", M.shape[1]),
            ("C", C),
            ("epsilon", epsilon),
            ("gamma", gamma),
            ("support vectors", int(len(support_idx))),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "alpha": alpha,
            "support_indices": support_idx,
            "intercept": intercept,
            "se": se,
            "n": n,
            "method": method_used,
        },
    )


def cheatsheet():
    return "svmge: SVM (epsilon-SVR) genomic predictor"


# CANONICAL TEST
# np.random.seed(12); M = np.random.randn(25, 4); y = np.sin(M[:,0]) + 0.2*np.random.randn(25)
# r = svm_genomic(np.zeros(25), y, M); residual SE small.
