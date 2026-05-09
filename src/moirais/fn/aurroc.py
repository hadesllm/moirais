# moirais.fn — function file (hadesllm/moirais)
"""Area under ROC curve with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from sklearn.metrics import roc_auc_score


def aurroc(y_true: Union[Sequence, np.ndarray],
           score: Union[Sequence, np.ndarray]):
    """Area under the ROC curve.

    Range [0, 1]; 0.5 = chance, 1 = perfect, 0 = perfectly inverted.

    Returns RichResult; float(result) yields the AUROC scalar.
    """
    from ._richresult import RichResult
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, s))
    if auc >= 0.9: bench = "excellent"
    elif auc >= 0.8: bench = "good"
    elif auc >= 0.7: bench = "fair"
    elif auc > 0.5: bench = "poor"
    else: bench = "no better than chance"
    warnings = []
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if min(pos, neg) < 10:
        warnings.append(f"class imbalance: {pos} positives, {neg} negatives - "
                        "AUROC unstable; consider stratified bootstrap CI.")
    return RichResult(
        title="Area under ROC curve",
        summary_lines=[
            ("AUROC", auc), ("Benchmark", bench),
            ("n total", len(y)),
            ("n positive (y=1)", pos), ("n negative (y=0)", neg),
        ],
        warnings=warnings,
        interpretation=f"AUROC={auc:.3f} -> {bench}.",
        payload={"value": auc, "statistic": auc, "benchmark": bench,
                 "n_positive": pos, "n_negative": neg},
    )
