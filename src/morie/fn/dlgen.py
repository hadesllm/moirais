# morie.fn — function file (hadesllm/morie)
"""Deep-learning genomic prediction (single-hidden-layer MLP, NumPy)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["deep_learning_genomic"]


def deep_learning_genomic(x, y, markers, hidden: int = 16, n_epochs: int = 200,
                          lr: float = 1e-2, l2: float = 1e-3, seed: int = 0,
                          deterministic_seed: int | None = None):
    """Single-hidden-layer MLP genomic predictor — NumPy implementation.

    Architecture::

        h = tanh(M @ W1 + b1)       # (n, hidden)
        y_hat = h @ w2 + b2 + X@gamma + intercept

    Trained by full-batch gradient descent with L2 weight decay. We use a
    small fixed-iteration / fixed-architecture configuration so behaviour is
    deterministic and reproducible.

    Parameters
    ----------
    x : array-like (n,) or (n,q). Optional fixed-effect design.
    y : array-like (n,)
    markers : array-like (n, m)
    hidden : int, default 16
    n_epochs : int, default 200
    lr : float, default 1e-2. Learning rate.
    l2 : float, default 1e-3. Weight-decay (ridge on W1/w2).
    seed : int
    deterministic_seed : int or None, optional
        If supplied, RNG state is derived from the SHA-keyed
        :func:`morie._det_rng.from_seed` so Py<->R streams agree for the
        canonical fixture.  When ``None`` (default), behaviour is
        unchanged.

    Returns
    -------
    RichResult with payload keys estimate, y_hat, beta (=gamma), W1, b1, w2,
    b2, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 12.
    """
    if deterministic_seed is not None:
        from morie._det_rng import from_seed
        rng = from_seed("dlgen", deterministic_seed)
    else:
        rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    m = M.shape[1]
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1 and Xa.size > 0:
        Xa = Xa.reshape(-1, 1)
    if Xa.size == 0:
        Xa = np.zeros((n, 0))
    q = Xa.shape[1] if Xa.size else 0
    # Standardise markers for stable training
    M_mean = M.mean(axis=0); M_sd = M.std(axis=0)
    M_sd = np.where(M_sd > 0, M_sd, 1.0)
    Ms = (M - M_mean) / M_sd
    # Init
    W1 = rng.normal(0, 1.0 / np.sqrt(m), size=(m, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 1.0 / np.sqrt(hidden), size=hidden)
    b2 = float(np.mean(y))
    gamma = np.zeros(q) if q else np.zeros(0)
    losses = []
    for _ in range(n_epochs):
        # Forward
        z1 = Ms @ W1 + b1
        h = np.tanh(z1)
        y_hat = h @ w2 + b2 + (Xa @ gamma if q else 0.0)
        resid = y_hat - y
        # Backward
        dy = resid / n
        dw2 = h.T @ dy + l2 * w2
        db2 = float(np.sum(dy))
        dh = np.outer(dy, w2)
        dz1 = dh * (1.0 - h ** 2)
        dW1 = Ms.T @ dz1 + l2 * W1
        db1 = dz1.sum(axis=0)
        if q:
            dgamma = Xa.T @ dy
        # Step
        W1 -= lr * dW1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        if q:
            gamma -= lr * dgamma
        losses.append(float(np.mean(resid ** 2)))
    # Final predictions
    h = np.tanh(Ms @ W1 + b1)
    y_hat = h @ w2 + b2 + (Xa @ gamma if q else 0.0)
    resid = y - y_hat
    se = float(np.sqrt(np.mean(resid ** 2)))
    return RichResult(
        title="Deep-learning genomic predictor (MLP-1H)",
        summary_lines=[
            ("n", n),
            ("m (markers)", m),
            ("hidden units", hidden),
            ("epochs", n_epochs),
            ("final train MSE", losses[-1] if losses else float("nan")),
            ("residual SE", se),
        ],
        payload={
            "estimate": float(np.mean(y_hat)),
            "y_hat": y_hat,
            "beta": gamma,
            "W1": W1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "loss_curve": np.asarray(losses),
            "se": se,
            "n": n,
            "method": "Single-hidden-layer MLP (NumPy)",
        },
        warnings=["NumPy implementation: for production-grade DL use "
                  "keras/torch on GPU."],
    )


def cheatsheet():
    return "dlgen: Deep-learning genomic predictor (MLP)"


# CANONICAL TEST
# np.random.seed(6); M = np.random.randn(20, 5); y = M[:,0] + 0.3*np.random.randn(20)
# r = deep_learning_genomic(np.zeros(20), y, M, seed=6); loss decreases.
