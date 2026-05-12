# morie.fn -- function file (hadesllm/morie)
"""1D Convolutional Neural Network for biosignal classification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cnnbs_fn(
    X: np.ndarray, y: np.ndarray, n_epochs: int = 10, n_filters: int = 8, kernel_size: int = 5
) -> DescriptiveResult:
    """Train a minimal 1D CNN for biosignal classification.

    :param X: Feature matrix (samples x time_steps).
    :param y: Class labels (integer-encoded).
    :param n_epochs: Training epochs (default 10).
    :param n_filters: Number of conv filters (default 8).
    :param kernel_size: Filter width (default 5).
    :return: DescriptiveResult with accuracy and learned filters.
    """
    from morie._classify import cnn_biosignal

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    filters, accuracy = cnn_biosignal(X, y, n_epochs=n_epochs, n_filters=n_filters, kernel_size=kernel_size)
    return DescriptiveResult(
        name="cnn_biosignal",
        value=accuracy,
        extra={"filters": filters, "accuracy": accuracy, "n_epochs": n_epochs},
    )


cnnbs = cnnbs_fn


def cheatsheet() -> str:
    return "cnnbs_fn({}) -> 1D Convolutional Neural Network for biosignal classification"
