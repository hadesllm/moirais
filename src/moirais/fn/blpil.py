# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"""

from __future__ import annotations

import pandas as pd

from ._containers import DescriptiveResult


def naive_baseline(
    data: pd.DataFrame,
    *,
    target: str = "outcome",
    method: str = "mean",
) -> DescriptiveResult:
    """Naive baseline predictor for benchmarking ML models.

    Returns the constant prediction that any ML model should beat:
    - ``"mean"``: predicts the training mean (regression).
    - ``"median"``: predicts the training median (robust regression).
    - ``"mode"``: predicts the most frequent class (classification).
    - ``"prior"``: predicts class proportions (probabilistic baseline).

    Parameters
    ----------
    data : DataFrame
        Training data.
    target : str
        Target column name.
    method : str
        Baseline strategy.

    Returns
    -------
    DescriptiveResult
        ``value`` is the baseline prediction; ``extra`` has baseline MSE
        or accuracy.
    """
    if target not in data.columns:
        raise ValueError(f"Column '{target}' not found")
    y = data[target].dropna()
    if len(y) == 0:
        raise ValueError("No non-null values in target")

    if method == "mean":
        pred = float(y.mean())
        mse = float(((y - pred) ** 2).mean())
        return DescriptiveResult(
            name="Naive Baseline (mean)",
            value=pred,
            extra={"mse": mse, "n": len(y)},
        )
    elif method == "median":
        pred = float(y.median())
        mae = float((y - pred).abs().mean())
        return DescriptiveResult(
            name="Naive Baseline (median)",
            value=pred,
            extra={"mae": mae, "n": len(y)},
        )
    elif method == "mode":
        mode_val = y.mode().iloc[0]
        accuracy = float((y == mode_val).mean())
        return DescriptiveResult(
            name="Naive Baseline (mode)",
            value=mode_val,
            extra={"accuracy": accuracy, "n": len(y)},
        )
    elif method == "prior":
        props = y.value_counts(normalize=True).to_dict()
        majority_acc = max(props.values())
        return DescriptiveResult(
            name="Naive Baseline (prior)",
            value=props,
            extra={"majority_accuracy": majority_acc, "n": len(y)},
        )
    else:
        raise ValueError(f"method must be 'mean', 'median', 'mode', or 'prior', got '{method}'")


blpil = naive_baseline


def cheatsheet() -> str:
    return "It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"
