# moirais.fn — function file (hadesllm/moirais)
"""Youden's J / informedness with R-style verbose result."""


def yindex(tpr: float, fpr: float):
    """Youden's J statistic: TPR - FPR.

    Maximized over thresholds gives optimal cutoff under equal costs.

    Returns RichResult; float(result) yields the scalar.
    """
    from ._richresult import RichResult
    if not 0 <= tpr <= 1 or not 0 <= fpr <= 1:
        raise ValueError(f"TPR ({tpr}) and FPR ({fpr}) must be in [0, 1].")
    j = tpr - fpr
    return RichResult(
        title="Youden's J statistic",
        summary_lines=[
            ("J = TPR - FPR", j), ("TPR (sensitivity)", tpr),
            ("FPR (1 - specificity)", fpr), ("Specificity", 1 - fpr),
        ],
        interpretation=(f"J={j:+.3f}; J=1 means perfect classifier, J=0 means "
                        "no better than chance, J<0 means inverse predictions."),
        payload={"value": j, "statistic": j, "tpr": tpr, "fpr": fpr},
    )
