"""SVM with kernel trick (RBF / poly / sigmoid)."""
import numpy as np

from ._richresult import RichResult

__all__ = ["svm_kernel_trick"]


def svm_kernel_trick(x, y, *, kernel="rbf", C=1.0, gamma="scale", degree=3, seed=0):
    """Kernel SVM via sklearn.svm.SVC.

    K(x_i, x_j) -- RBF: exp(-gamma ||x_i - x_j||^2);
    poly: (gamma <x_i, x_j> + coef0)^degree; sigmoid: tanh(gamma <x_i,x_j>+coef0).

    Parameters
    ----------
    x : array-like (n, p).
    y : array-like (n,) -- binary labels.
    kernel : {"rbf", "poly", "sigmoid", "linear"}.
    C : float
        Soft-margin regularization inverse.
    gamma : "scale" | "auto" | float.
    degree : int
        Degree for "poly" kernel.
    seed : int
        random_state for libsvm.

    Returns
    -------
    RichResult with payload: estimate (train accuracy), n_support (per class),
    kernel, C, gamma, n, method.
    """
    from sklearn.svm import SVC

    X = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=seed)
    clf.fit(X, y)
    acc = float(clf.score(X, y))
    return RichResult(payload={
        "estimate": acc,
        "train_accuracy": acc,
        "n_support": clf.n_support_.tolist(),
        "kernel": kernel,
        "C": float(C),
        "gamma": str(gamma),
        "degree": int(degree),
        "n": int(n),
        "method": f"Kernel SVM ({kernel})",
    })


def cheatsheet():
    return "svmkr: kernel SVM (RBF/poly/sigmoid)"


# CANONICAL TEST
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.0).astype(int)
    r = svm_kernel_trick(X, y, kernel="rbf", C=1.0)
    print("kernel:", r.kernel, "  C:", r.C, "  gamma:", r.gamma)
    print("train accuracy:", r.train_accuracy)
    print("n_support per class:", r.n_support)
