# morie.fn -- function file (hadesllm/morie)
"""Power for two-sample t-test with R-style verbose result."""

from scipy.stats import nct, t as _t


def powtt2(d: float, n1: int, n2: int, alpha: float = 0.05):
    """Power for two-sample t-test of mean difference at effect d."""
    from ._richresult import RichResult
    if n1 < 2 or n2 < 2 or d == 0:
        raise ValueError(f"require n1,n2>=2 and d!=0; got n1={n1}, n2={n2}, d={d}.")
    df = n1 + n2 - 2
    nc = d * (n1 * n2 / (n1 + n2)) ** 0.5
    crit = _t.ppf(1 - alpha / 2, df)
    pw = float(1 - nct.cdf(crit, df, nc) + nct.cdf(-crit, df, nc))
    return RichResult(
        title="Power for two-sample t-test",
        summary_lines=[
            ("Power (1 - beta)", pw),
            ("Type II error (beta)", 1 - pw),
            ("Effect size d", d),
            ("n group 1", n1), ("n group 2", n2),
            ("alpha (two-sided)", alpha),
            ("df", df),
        ],
        warnings=[] if pw >= 0.8 else
                 [f"power={pw:.2f} < 0.80 conventional minimum; consider larger n "
                  "or use `npowtt` to size for power=0.80."],
        interpretation=(f"At d={d}, n1={n1}, n2={n2}, alpha={alpha}: probability of "
                        f"detecting effect = {pw*100:.1f}%."),
        payload={"value": pw, "statistic": pw, "beta": 1 - pw, "df": df},
    )
