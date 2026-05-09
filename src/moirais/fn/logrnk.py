# moirais.fn — function file (hadesllm/moirais)
"""Log-rank test (two-group survival) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import chi2


def logrnk(times1, events1, times2, events2):
    """Log-rank test for difference between two survival curves."""
    from ._richresult import hypothesis_test_result
    t1 = np.asarray(times1, dtype=float); e1 = np.asarray(events1, dtype=int)
    t2 = np.asarray(times2, dtype=float); e2 = np.asarray(events2, dtype=int)
    all_t = np.concatenate([t1, t2])
    all_e = np.concatenate([e1, e2])
    grp = np.concatenate([np.zeros_like(t1), np.ones_like(t2)])
    obs1 = exp1 = var = 0.0
    for ut in np.unique(all_t):
        d1 = np.sum((all_t == ut) & (all_e == 1) & (grp == 0))
        d2 = np.sum((all_t == ut) & (all_e == 1) & (grp == 1))
        d = d1 + d2
        if d == 0:
            continue
        n_at_1 = np.sum(t1 >= ut); n_at_2 = np.sum(t2 >= ut)
        n_at = n_at_1 + n_at_2
        if n_at < 2 or n_at_1 == 0 or n_at_2 == 0:
            continue
        e1_t = d * n_at_1 / n_at
        v_t = (n_at_1 * n_at_2 * d * (n_at - d)) / (n_at * n_at * (n_at - 1))
        obs1 += d1; exp1 += e1_t; var += v_t
    if var == 0:
        chi = 0.0; p = 1.0
    else:
        chi = (obs1 - exp1) ** 2 / var
        p = float(1 - chi2.cdf(chi, 1))
    return hypothesis_test_result(
        test_name="Log-rank test (two-group survival)",
        statistic=float(chi), df=1, pvalue=p,
        extra_summary=[("n group 1", int(t1.size)), ("n group 2", int(t2.size)),
                       ("Observed events grp 1", int(obs1)),
                       ("Expected events grp 1", float(exp1)),
                       ("Observed events grp 2", int(np.sum(e2 == 1)))],
        warnings=[] if (t1.size + t2.size) >= 20 else
                 ["small total n; chi-squared approximation may be poor."],
    )

# Back-compat alias
logrank_test = logrnk
