# morie.fn — function file (hadesllm/morie)
"""Recurrent (vanilla RNN) genomic predictor — NumPy."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rnn_genomic"]


def rnn_genomic(x, y, markers, hidden: int = 8, n_epochs: int = 150,
                lr: float = 1e-2, l2: float = 1e-3, seed: int = 0):
    """Vanilla RNN sweeping over the marker sequence.

    Architecture (per individual i with marker vector m_i of length L)::

        h_0 = 0
        for t = 1..L:
            h_t = tanh(W_h h_{t-1} + W_x * m_i[t] + b_h)
        y_hat_i = w_o^T h_L + b_o

    Trained by backpropagation through time, full-batch GD with L2 decay.
    Used in the Montesinos book as a baseline before LSTM/GRU — we keep
    the vanilla version because it's deterministic and trains in
    sub-second on small inputs.

    Parameters
    ----------
    x, y, markers : standard.
    hidden : int, default 8
    n_epochs, lr, l2, seed : training hyperparameters

    Returns
    -------
    RichResult with payload keys estimate, y_hat, W_h, W_x, w_o, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 14.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    L = M.shape[1]
    M_mean = M.mean(axis=0); M_sd = M.std(axis=0)
    M_sd = np.where(M_sd > 0, M_sd, 1.0)
    Ms = (M - M_mean) / M_sd
    H = hidden
    Wh = rng.normal(0, 1.0 / np.sqrt(H), size=(H, H))
    Wx = rng.normal(0, 1.0, size=H)
    bh = np.zeros(H)
    wo = rng.normal(0, 1.0 / np.sqrt(H), size=H)
    bo = float(np.mean(y))
    losses = []
    for _ in range(n_epochs):
        # Forward all individuals at once
        h = np.zeros((n, H))
        hs = [h.copy()]
        for t in range(L):
            xt = Ms[:, t][:, None]   # (n, 1)
            h = np.tanh(h @ Wh + xt * Wx + bh)
            hs.append(h.copy())
        y_hat = h @ wo + bo
        resid = y_hat - y
        dy = resid / n
        # Backward
        dwo = h.T @ dy + l2 * wo
        dbo = float(dy.sum())
        dh = np.outer(dy, wo)
        dWh = np.zeros_like(Wh)
        dWx = 0.0
        dbh = np.zeros(H)
        for t in reversed(range(L)):
            h_t = hs[t + 1]
            dh_raw = dh * (1.0 - h_t ** 2)
            dWh += hs[t].T @ dh_raw
            dWx += float((Ms[:, t][:, None] * dh_raw).sum())
            dbh += dh_raw.sum(axis=0)
            dh = dh_raw @ Wh.T
        dWh += l2 * Wh
        # Step
        Wh -= lr * dWh
        Wx -= lr * (dWx + l2 * Wx)
        bh -= lr * dbh
        wo -= lr * dwo
        bo -= lr * dbo
        losses.append(float(np.mean(resid ** 2)))
    # Final
    h = np.zeros((n, H))
    for t in range(L):
        xt = Ms[:, t][:, None]
        h = np.tanh(h @ Wh + xt * Wx + bh)
    y_hat = h @ wo + bo
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="RNN genomic predictor (vanilla)",
        summary_lines=[
            ("n", n),
            ("L (markers / steps)", L),
            ("hidden", H),
            ("epochs", n_epochs),
            ("final train MSE", losses[-1] if losses else float("nan")),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "W_h": Wh, "W_x": float(Wx) if np.ndim(Wx) == 0 else Wx,
            "b_h": bh, "w_o": wo, "b_o": bo,
            "loss_curve": np.asarray(losses),
            "se": se,
            "n": n,
            "method": "Vanilla RNN BPTT (NumPy)",
        },
        warnings=["Vanilla RNN: vanishing gradients on long L. For "
                  "production use LSTM/GRU in keras/torch."],
    )


def cheatsheet():
    return "rnnge: RNN/LSTM genomic predictor"


# CANONICAL TEST
# np.random.seed(8); M = np.random.randn(15, 6); y = np.sum(M, axis=1) + 0.2*np.random.randn(15)
# r = rnn_genomic(np.zeros(15), y, M, seed=8); loss decreases.
