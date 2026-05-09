# moirais.fn — function file (hadesllm/moirais)
"""Brier score (calibration loss) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def brierl(p_pred: Union[Sequence, np.ndarray],
           y_true: Union[Sequence, np.ndarray]):
    """Brier score: (1/n) Sigma (p_i - y_i)^2.

    Lower is better. Range [0, 1]. Decomposable into reliability +
    resolution + uncertainty.

    Returns RichResult; float(result) yields the scalar.
    """
    from ._richresult import RichResult
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}.")
    bs = float(np.mean((p - y) ** 2))
    base_rate = float(y.mean())
    bs_baseline = float(np.mean((base_rate - y) ** 2))
    skill = 1 - bs / bs_baseline if bs_baseline > 0 else float("nan")
    return RichResult(
        title="Brier score (calibration loss)",
        summary_lines=[
            ("Brier score", bs),
            ("Baseline (predict base rate)", bs_baseline),
            ("Brier skill score", skill),
            ("n", int(p.size)), ("Base rate (mean y)", base_rate),
        ],
        warnings=[] if 0 <= base_rate <= 1 else
                 ["y_true has values outside [0,1] - check encoding."],
        interpretation=(
            f"BS={bs:.4f}; baseline (predict {base_rate:.3f} for everyone) = "
            f"{bs_baseline:.4f}. Brier skill score {skill:+.3f} (positive = "
            f"better than baseline)."
        ),
        payload={"value": bs, "statistic": bs, "skill_score": skill,
                 "baseline": bs_baseline},
    )
