# morie.fn — function file (hadesllm/morie)
"""LSTM network for biosignal classification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def lstmb_fn(X: np.ndarray, y: np.ndarray, n_epochs: int = 10, hidden_size: int = 16) -> DescriptiveResult:
    """Train a minimal LSTM for biosignal sequence classification.

    :param X: Feature matrix (samples x time_steps).
    :param y: Class labels (integer-encoded).
    :param n_epochs: Training epochs (default 10).
    :param hidden_size: LSTM hidden state dimension (default 16).
    :return: DescriptiveResult with accuracy and model parameters.
    """
    from morie._classify import lstm_biosignal

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    params, accuracy = lstm_biosignal(X, y, n_epochs=n_epochs, hidden_size=hidden_size)
    return DescriptiveResult(
        name="lstm_biosignal",
        value=accuracy,
        extra={"accuracy": accuracy, "hidden_size": hidden_size, "n_epochs": n_epochs},
    )


lstmb = lstmb_fn


def cheatsheet() -> str:
    return "lstmb_fn({}) -> LSTM network for biosignal classification."
