# morie.fn — function file (hadesllm/morie)
"""Matthews correlation coefficient with R-style verbose result."""


def mcc(tp: int, tn: int, fp: int, fn: int):
    """Matthews correlation coefficient for 2x2 confusion matrix.

    Balanced even with imbalanced classes; range [-1, 1].

    Returns RichResult; float(result) yields the scalar.
    """
    from ._richresult import RichResult
    num = tp * tn - fp * fn
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        v = 0.0
        warnings = ["denominator is zero - MCC defined as 0; check class imbalance."]
    else:
        v = float(num / denom)
        warnings = []
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    return RichResult(
        title="Matthews correlation coefficient",
        summary_lines=[
            ("MCC", v),
            ("Accuracy (for context)", accuracy),
            ("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn),
            ("n total", n),
        ],
        warnings=warnings,
        interpretation=(f"MCC={v:+.3f}: {abs(v):.2f} magnitude on [0, 1] scale; "
                        "negative = predictions inversely correlated with truth."),
        payload={"value": v, "statistic": v, "tp": tp, "tn": tn, "fp": fp, "fn": fn},
    )

# Back-compat alias
matthews_corrcoef = mcc
