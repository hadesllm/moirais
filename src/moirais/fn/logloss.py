# moirais.fn — function file (hadesllm/moirais)
"""Cross-entropy / log loss with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def logloss(p_pred: Union[Sequence, np.ndarray],
            y_true: Union[Sequence, np.ndarray],
            eps: float = 1e-15):
    """Binary cross-entropy: -1/n Sigma [y_i log p_i + (1-y_i) log(1-p_i)].

    Lower is better. 0 = perfect calibration.
    """
    from ._richresult import RichResult
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}.")
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base_rate = float(y.mean())
    if base_rate <= eps or base_rate >= 1 - eps:
        ll_baseline = float("nan")
        skill = float("nan")
    else:
        ll_baseline = float(-(base_rate * np.log(base_rate) +
                              (1 - base_rate) * np.log(1 - base_rate)))
        skill = 1 - ll / ll_baseline if ll_baseline > 0 else float("nan")
    return RichResult(
        title="Binary cross-entropy / log loss",
        summary_lines=[
            ("Log loss", ll), ("Baseline (predict base rate)", ll_baseline),
            ("Log-loss skill score", skill),
            ("n", int(p.size)), ("Base rate (mean y)", base_rate),
            ("eps clip", eps),
        ],
        interpretation=(
            f"log loss = {ll:.4f}; baseline = {ll_baseline:.4f}; skill "
            f"score {skill:+.3f} (positive = beats baseline)."
        ),
        payload={"value": ll, "statistic": ll, "skill_score": skill,
                 "baseline": ll_baseline},
    )
