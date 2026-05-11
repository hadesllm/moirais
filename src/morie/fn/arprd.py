# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""AR model multi-step ahead prediction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ar_predict_fn(x: np.ndarray, order: int = 10, n_ahead: int = 1) -> DescriptiveResult:
    """Predict future values using an estimated AR model.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :param n_ahead: Number of steps ahead to predict (default 1).
    :return: DescriptiveResult with predictions array in extra.
    """
    from morie._armodel import ar_predict, ar_yule_walker

    x = np.asarray(x, dtype=float).ravel()
    a, _ = ar_yule_walker(x, order=order)
    predictions = ar_predict(x, a, n_ahead=n_ahead)
    return DescriptiveResult(
        name="ar_predict",
        value=float(predictions[-1]) if len(predictions) == 1 else None,
        extra={"predictions": predictions, "n_ahead": n_ahead},
    )


arprd = ar_predict_fn


def cheatsheet() -> str:
    return "ar_predict_fn({}) -> AR model multi-step ahead prediction."
