"""SVM primal with hinge loss (linear kernel)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["svm_hinge_primal"]


def svm_hinge_primal(x, y, *, C=1.0, seed=0):
    """Linear SVM (primal hinge loss) via sklearn.svm.LinearSVC.

    Solves min (1/2)||w||^2 + C sum_i max(0, 1 - y_i (w'x_i + b)).
    Labels are coerced to {-1, +1} internally.

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,) — any binary labels.
    C : float
        Regularization inverse (smaller C => stronger regularization).
    seed : int
        random_state for the optimizer.

    Returns
    -------
    RichResult with payload: estimate (intercept then w), accuracy
    (training accuracy), C, n, method.
    """
    from sklearn.svm import LinearSVC

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"svmhg requires binary y, got {len(classes)} classes")
    # Coerce to -1/+1 for clarity in coefficients
    y_b = np.where(y == classes[1], 1, -1)
    clf = LinearSVC(C=C, loss="hinge", max_iter=20000, random_state=seed,
                    dual=True)
    clf.fit(X, y_b)
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    acc = float(clf.score(X, y_b))
    return RichResult(payload={
        "estimate": [b] + w.tolist(),
        "intercept": b,
        "weights": w.tolist(),
        "train_accuracy": acc,
        "C": float(C),
        "classes": classes.tolist(),
        "n": int(n),
        "method": "Linear SVM (primal hinge loss)",
    })


def cheatsheet():
    return "svmhg: linear SVM with hinge loss"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    r = svm_hinge_primal(X, y, C=1.0)
    print("intercept:", r.intercept, "weights:", r.weights)
    print("train accuracy:", r.train_accuracy)
