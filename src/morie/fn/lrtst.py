# morie.fn -- function file (hadesllm/morie)
"""Likelihood ratio test with R-style verbose result."""

from scipy.stats import chi2


def lrtst(loglik_full: float, loglik_reduced: float, df_diff: int):
    """LR test for nested models. LR = -2(LL_red - LL_full) ~ chi^2(df_diff)."""
    from ._richresult import hypothesis_test_result
    if df_diff <= 0:
        raise ValueError(f"df_diff must be positive, got {df_diff}.")
    if loglik_full < loglik_reduced:
        raise ValueError(f"LL(full)={loglik_full} should be >= LL(reduced)={loglik_reduced}.")
    lr = -2.0 * (loglik_reduced - loglik_full)
    p = 1.0 - chi2.cdf(lr, df=df_diff)
    return hypothesis_test_result(
        test_name="Likelihood ratio test",
        statistic=float(lr), df=df_diff, pvalue=float(p),
        extra_summary=[
            ("LL(full)", loglik_full), ("LL(reduced)", loglik_reduced),
            ("LL difference", loglik_full - loglik_reduced),
            ("Extra parameters", df_diff),
        ],
    )
