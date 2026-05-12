# morie.fn — function file (hadesllm/morie)
"""CNN for genomic prediction — NumPy Conv1D + dense, deterministic."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["cnn_genomic"]


def _conv1d(M, W, b, stride: int = 1):
    """Single-channel 1D valid convolution with bias."""
    n, m = M.shape
    k, f = W.shape
    out_len = (m - k) // stride + 1
    out = np.zeros((n, out_len, f))
    for s in range(out_len):
        seg = M[:, s * stride:s * stride + k]
        out[:, s, :] = seg @ W + b
    return out


def cnn_genomic(x, y, markers, n_filters: int = 8, kernel: int = 3,
                hidden: int = 8, n_epochs: int = 150, lr: float = 1e-2,
                l2: float = 1e-3, seed: int = 0):
    """1D convolutional genomic predictor (NumPy).

    Architecture::

        z = Conv1D(M; W_conv, b_conv)   # (n, L, n_filters)
        a = ReLU(z)                     # element-wise
        p = mean(a, axis=L)             # global-average pool -> (n, n_filters)
        h = tanh(p @ W1 + b1)           # (n, hidden)
        y_hat = h @ w2 + b2

    Trained by full-batch gradient descent with weight decay.

    Parameters
    ----------
    x, y, markers : see deep_learning_genomic.
    n_filters, kernel, hidden, n_epochs, lr, l2, seed : hyperparameters.

    Returns
    -------
    RichResult with payload keys estimate, y_hat, W_conv, W1, w2, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 13.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    m = M.shape[1]
    if kernel > m:
        kernel = max(1, m)
    # Standardise
    M_mean = M.mean(axis=0); M_sd = M.std(axis=0)
    M_sd = np.where(M_sd > 0, M_sd, 1.0)
    Ms = (M - M_mean) / M_sd
    Wc = rng.normal(0, 1.0 / np.sqrt(kernel), size=(kernel, n_filters))
    bc = np.zeros(n_filters)
    W1 = rng.normal(0, 1.0 / np.sqrt(n_filters), size=(n_filters, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=hidden)
    b2 = float(np.mean(y))

    losses = []
    for _ in range(n_epochs):
        z = _conv1d(Ms, Wc, bc)          # (n, L, f)
        a = np.maximum(z, 0)             # ReLU
        p = a.mean(axis=1)               # (n, f)
        h_pre = p @ W1 + b1
        h = np.tanh(h_pre)
        y_hat = h @ w2 + b2
        resid = y_hat - y
        dy = resid / n
        # Dense backward
        dw2 = h.T @ dy + l2 * w2
        db2 = float(dy.sum())
        dh = np.outer(dy, w2)
        dh_pre = dh * (1.0 - h ** 2)
        dW1 = p.T @ dh_pre + l2 * W1
        db1 = dh_pre.sum(axis=0)
        dp = dh_pre @ W1.T              # (n, f)
        # Backprop through mean-pool
        L = a.shape[1]
        da = np.repeat(dp[:, None, :], L, axis=1) / L
        dz = da * (z > 0)
        # Conv1D backward
        dWc = np.zeros_like(Wc)
        dbc = dz.sum(axis=(0, 1))
        for s in range(L):
            seg = Ms[:, s:s + kernel]
            dWc += seg.T @ dz[:, s, :]
        dWc += l2 * Wc
        # Step
        Wc -= lr * dWc; bc -= lr * dbc
        W1 -= lr * dW1; b1 -= lr * db1
        w2 -= lr * dw2; b2 -= lr * db2
        losses.append(float(np.mean(resid ** 2)))
    # Final
    z = _conv1d(Ms, Wc, bc)
    a = np.maximum(z, 0); p = a.mean(axis=1)
    h = np.tanh(p @ W1 + b1)
    y_hat = h @ w2 + b2
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="CNN genomic predictor (Conv1D + GAP + dense)",
        summary_lines=[
            ("n", n),
            ("m (markers)", m),
            ("n_filters", n_filters),
            ("kernel", kernel),
            ("epochs", n_epochs),
            ("final train MSE", losses[-1] if losses else float("nan")),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "W_conv": Wc,
            "b_conv": bc,
            "W1": W1, "b1": b1, "w2": w2, "b2": b2,
            "loss_curve": np.asarray(losses),
            "se": se,
            "n": n,
            "method": "Conv1D genomic predictor (NumPy)",
        },
        warnings=["NumPy CNN: small kernel + GAP for stability; use "
                  "keras/torch for production."],
    )


def cheatsheet():
    return "cnnge: CNN genomic predictor"


# CANONICAL TEST
# np.random.seed(7); M = np.random.randn(20, 8); y = M[:,1]+M[:,3]+0.2*np.random.randn(20)
# r = cnn_genomic(np.zeros(20), y, M, seed=7); loss decreases over epochs.
