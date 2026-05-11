# morie.fn — function file (hadesllm/morie)
"""Required n for two-sample t-test power with R-style verbose result."""

from scipy.stats import nct, t as _t


def npowtt(d: float, target_power: float = 0.8,
           alpha: float = 0.05, max_n: int = 10000):
    """Smallest equal-group n giving target power for two-sample t."""
    from ._richresult import RichResult
    if d == 0 or not 0 < target_power < 1:
        raise ValueError(f"invalid d ({d}) or target_power ({target_power}).")
    found = None
    for n in range(2, max_n + 1):
        df = 2 * n - 2
        nc = d * (n / 2) ** 0.5
        crit = _t.ppf(1 - alpha / 2, df)
        pw = 1 - nct.cdf(crit, df, nc) + nct.cdf(-crit, df, nc)
        if pw >= target_power:
            found = (n, pw)
            break
    if found is None:
        n, pw = max_n, float(pw)
    else:
        n, pw = found
        pw = float(pw)
    return RichResult(
        title="Required n for two-sample t-test power",
        summary_lines=[
            ("n per group", n),
            ("Total n", 2 * n),
            ("Achieved power", pw),
            ("Target power", target_power),
            ("Effect size d", d), ("alpha", alpha),
        ],
        warnings=[] if found else
                 [f"target power not achieved within max_n={max_n}; effect "
                  "is too small relative to noise."],
        interpretation=(f"For d={d}, alpha={alpha}: need n={n} per group "
                        f"({2*n} total) for power >={target_power}."),
        payload={"value": n, "statistic": n, "n_per_group": n,
                 "achieved_power": pw},
    )
