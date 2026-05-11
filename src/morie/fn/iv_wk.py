# morie.fn — function file (hadesllm/morie)
"""Weak instrument diagnostics for instrumental variables."""

from __future__ import annotations

from typing import Any


def weak_instrument_test(
    first_stage_F: float,
    *,
    n_instruments: int = 1,
    n_endogenous: int = 1,
) -> dict[str, Any]:
    """
    Diagnose weak instruments using the first-stage F-statistic.

    Rule of thumb (Staiger & Stock, 1997): F < 10 indicates weak
    instruments. Stock & Yogo (2005) provide critical values for specific
    bias thresholds.

    For a single endogenous regressor with L instruments, the Stock-Yogo
    5% relative bias critical values (approximate) are:

    ====  =========
    L     CV (5%)
    ====  =========
    1     N/A (just-identified)
    2     19.93
    3     22.30
    4     24.58
    5     26.87
    ====  =========

    :param first_stage_F: F-statistic from the first-stage regression.
    :param n_instruments: Number of excluded instruments (L).
    :param n_endogenous: Number of endogenous regressors (default 1).
    :return: Dictionary with F_stat, is_weak, stock_yogo_5pct,
        passes_stock_yogo.

    References
    ----------
    Staiger, D., & Stock, J. H. (1997). Instrumental variables regression
    with weak instruments. *Econometrica*, 65(3), 557--586.

    Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in
    linear IV regression. In D. W. K. Andrews & J. H. Stock (Eds.),
    *Identification and Inference for Econometric Models*. Cambridge UP.
    """
    # Stock-Yogo 5% relative bias CV for single endogenous regressor
    _sy_5pct = {2: 19.93, 3: 22.30, 4: 24.58, 5: 26.87, 6: 29.18, 7: 31.50, 8: 33.84, 9: 36.19, 10: 38.54}

    sy_cv = _sy_5pct.get(n_instruments) if n_endogenous == 1 else None

    is_weak_rule = first_stage_F < 10.0
    passes_sy = (first_stage_F >= sy_cv) if sy_cv is not None else None

    return {
        "F_stat": float(first_stage_F),
        "is_weak": is_weak_rule,
        "stock_yogo_5pct": sy_cv,
        "passes_stock_yogo": passes_sy,
        "n_instruments": n_instruments,
    }


iv_wk = weak_instrument_test


def cheatsheet() -> str:
    return "weak_instrument_test({}) -> Weak instrument diagnostics for instrumental variables."
