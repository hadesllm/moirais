"""ROC curve and AUC computation."""
import numpy as np

from ._richresult import RichResult

__all__ = ["roc_auc_score"]


def roc_auc_score(y_true, y_score):
    """ROC curve and AUC via sklearn.metrics.{roc_curve, roc_auc_score}.

    AUC = integral_0^1 TPR(FPR) dFPR.  Equivalent to the Mann-Whitney U
    probability that a positive instance receives a higher score than a
    negative one (Hanley & McNeil 1982).

    Binary y_true required; multi-class can be wrapped externally with
    one-vs-rest scoring.

    Parameters
    ----------
    y_true : array-like (n,)
        Binary labels (any two unique values; the larger is treated as positive).
    y_score : array-like (n,)
        Predicted scores or probabilities for the positive class.

    Returns
    -------
    RichResult with payload: estimate (AUC), auc, fpr, tpr, thresholds,
    n, n_positive, n_negative, method.
    """
    from sklearn.metrics import roc_auc_score as _roc_auc, roc_curve

    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    classes = np.unique(yt)
    if len(classes) != 2:
        raise ValueError(f"rocau requires binary y_true, got {len(classes)} classes")
    pos = classes[-1]
    yt_b = (yt == pos).astype(int)
    auc = float(_roc_auc(yt_b, ys))
    fpr, tpr, thr = roc_curve(yt_b, ys)
    return RichResult(payload={
        "estimate": auc,
        "auc": auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
        "n": int(len(yt)),
        "n_positive": int(np.sum(yt_b == 1)),
        "n_negative": int(np.sum(yt_b == 0)),
        "method": "ROC AUC (Mann-Whitney U probability)",
    })


def cheatsheet():
    return "rocau: ROC curve and AUC"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    # Score is correlated with truth via Gaussian noise
    s = y + rng.normal(scale=0.6, size=n)
    r = roc_auc_score(y, s)
    print("AUC:", r.auc)
    print("n_positive:", r.n_positive, "  n_negative:", r.n_negative)
    print("FPR[0:5]:", r.fpr[:5])
    print("TPR[0:5]:", r.tpr[:5])
