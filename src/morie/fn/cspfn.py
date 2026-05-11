# morie.fn — function file (hadesllm/morie)
"""Common Spatial Patterns for EEG classification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def csp_filter(X_class1, X_class2, n_components: int = 4, **kwargs) -> DescriptiveResult:
    """Common Spatial Patterns for two-class EEG classification.

    Finds spatial filters that maximise the variance ratio between
    two classes.

    Parameters
    ----------
    X_class1 : array-like, shape (n_trials, n_channels, n_samples)
        EEG trials for class 1.
    X_class2 : array-like, shape (n_trials, n_channels, n_samples)
        EEG trials for class 2.
    n_components : int
        Number of CSP components per class (default 4; total = 2*n_components).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of spatial filters; ``extra`` has ``filters``,
        ``eigenvalues``, ``features_class1``, ``features_class2``.

    References
    ----------
    Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000).
    Optimal spatial filtering of single trial EEG during imagined hand
    movement. *IEEE Trans. Rehab. Eng.*, 8(4), 441-446.
    """
    X1 = np.asarray(X_class1, dtype=float)
    X2 = np.asarray(X_class2, dtype=float)

    def _avg_cov(X):
        covs = []
        for trial in X:
            c = trial @ trial.T / trial.shape[1]
            covs.append(c / np.trace(c))
        return np.mean(covs, axis=0)

    C1 = _avg_cov(X1)
    C2 = _avg_cov(X2)
    Cc = C1 + C2
    eigvals_c, U = np.linalg.eigh(Cc)
    idx = np.argsort(eigvals_c)[::-1]
    eigvals_c = eigvals_c[idx]
    U = U[:, idx]
    P = np.diag(1.0 / np.sqrt(eigvals_c + 1e-10)) @ U.T
    S1 = P @ C1 @ P.T
    eigvals, W = np.linalg.eigh(S1)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    W = W[:, idx]
    filters_full = W.T @ P
    n_ch = filters_full.shape[0]
    k = min(n_components, n_ch // 2)
    sel = list(range(k)) + list(range(n_ch - k, n_ch))
    filters = filters_full[sel]

    def _extract_features(X, F):
        feats = []
        for trial in X:
            z = F @ trial
            var = np.var(z, axis=1)
            feats.append(np.log(var / (np.sum(var) + 1e-15)))
        return np.array(feats)

    f1 = _extract_features(X1, filters)
    f2 = _extract_features(X2, filters)
    return DescriptiveResult(
        name="csp_filter",
        value=len(sel),
        extra={"filters": filters, "eigenvalues": eigvals, "features_class1": f1, "features_class2": f2},
    )


cspfn = csp_filter


def cheatsheet() -> str:
    return "csp_filter({}) -> Common Spatial Patterns for EEG classification."
