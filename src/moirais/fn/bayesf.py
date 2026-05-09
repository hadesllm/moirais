# moirais.fn — function file (hadesllm/moirais)
"""Bayes factor (BIC approximation) with R-style verbose result."""

import math


def bayesf(loglik_h1: float, loglik_h0: float,
           k_h1: int, k_h0: int, n: int):
    """Bayes factor BF_10 approximated from BIC difference.

    BF_10 ~= exp(-(BIC_1 - BIC_0) / 2)
    Convention: BF > 3 favors H1, BF > 10 strong, BF > 100 decisive.
    """
    from ._richresult import RichResult
    if n < 1 or k_h1 < 0 or k_h0 < 0:
        raise ValueError("invalid n, k_h1, or k_h0.")
    bic_h1 = k_h1 * math.log(n) - 2 * loglik_h1
    bic_h0 = k_h0 * math.log(n) - 2 * loglik_h0
    bf = float(math.exp(-(bic_h1 - bic_h0) / 2))
    if bf > 100: strength = "decisive in favor of H1"
    elif bf > 30: strength = "very strong in favor of H1"
    elif bf > 10: strength = "strong in favor of H1"
    elif bf > 3: strength = "substantial in favor of H1"
    elif bf > 1: strength = "weak in favor of H1"
    elif bf > 0.33: strength = "weak in favor of H0"
    elif bf > 0.1: strength = "substantial in favor of H0"
    elif bf > 0.033: strength = "strong in favor of H0"
    elif bf > 0.01: strength = "very strong in favor of H0"
    else: strength = "decisive in favor of H0"
    return RichResult(
        title="Bayes factor (BIC approximation)",
        summary_lines=[
            ("BF_10", bf), ("log10(BF_10)", math.log10(bf) if bf > 0 else float("-inf")),
            ("Strength of evidence", strength),
            ("BIC(H1)", bic_h1), ("BIC(H0)", bic_h0),
            ("LL(H1)", loglik_h1), ("LL(H0)", loglik_h0),
            ("k(H1)", k_h1), ("k(H0)", k_h0), ("n", n),
        ],
        interpretation=f"BF_10 = {bf:.4g} -> {strength} (Jeffreys/Kass-Raftery scale).",
        payload={"value": bf, "statistic": bf, "strength": strength,
                 "bic_h1": bic_h1, "bic_h0": bic_h0},
    )
