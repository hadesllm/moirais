# moirais.fn — function file (hadesllm/moirais)
"""Durbin-Watson statistic with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def dwtest(residuals: Union[Sequence, np.ndarray]):
    """Durbin-Watson statistic for autocorrelated residuals.

    DW = Sigma(e_t - e_{t-1})^2 / Sigma e_t^2

    Range: 0 (positive autocorr) - 2 (none) - 4 (negative).
    """
    from ._richresult import RichResult
    e = np.asarray(residuals, dtype=float)
    if e.size < 2:
        raise ValueError(f"need at least 2 residuals, got {e.size}.")
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
    if dw < 1.5: verdict = "positive autocorrelation likely"
    elif dw > 2.5: verdict = "negative autocorrelation likely"
    else: verdict = "no clear evidence of autocorrelation"
    return RichResult(
        title="Durbin-Watson test",
        summary_lines=[
            ("DW statistic", dw),
            ("n residuals", int(e.size)),
            ("Verdict", verdict),
        ],
        interpretation=(
            f"DW={dw:.3f} -> {verdict}. Critical values depend on n and "
            "predictors; consult Durbin-Watson tables for formal test, or "
            "use Ljung-Box (`ljbox`) for an asymptotic alternative."
        ),
        payload={"value": dw, "statistic": dw, "verdict": verdict},
    )
