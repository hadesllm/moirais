# moirais.fn — function file (hadesllm/moirais)
"""Out-of-sample prediction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._containers import DescriptiveResult


def oos_predict(
    model_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> DescriptiveResult:
    """Compute out-of-sample predictions using a model function.

    The *model_fn* should accept (X_train, y_train, X_test) and return
    predicted values for X_test.

    :param model_fn: Callable that fits and predicts.
    :param X_train: Training features.
    :param y_train: Training targets.
    :param X_test: Test features.
    :return: DescriptiveResult with predictions in ``extra['predictions']``.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    preds = model_fn(X_train, y_train, X_test)
    preds = np.asarray(preds)

    return DescriptiveResult(
        name="oos_predict",
        value=float(np.mean(preds)),
        extra={"predictions": preds, "n_train": len(y_train), "n_test": X_test.shape[0]},
    )


def cheatsheet() -> str:
    return "oos_predict(model_fn, X_train, y_train, X_test) -> OOS predictions"


oospr = oos_predict
