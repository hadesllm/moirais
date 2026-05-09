# moirais.fn — function file (hadesllm/moirais)
"""Permutation p-value (two-sample) with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def permpv(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray],
           n_perm: int = 5000, seed: int = 42):
    """Permutation p-value for H0: same distribution.

    Test statistic: |mean(x) - mean(y)|. Two-sided.
    """
    from ._richresult import hypothesis_test_result
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    pooled = np.concatenate([a, b])
    n_a = a.size
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = abs(pooled[:n_a].mean() - pooled[n_a:].mean())
        if diff >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return hypothesis_test_result(
        test_name="Permutation test (two-sample mean difference)",
        statistic=float(obs), pvalue=float(p),
        extra_summary=[
            ("Observed |mean diff|", obs),
            ("Permutations", n_perm),
            ("Permutations >= observed", count),
            ("n(x)", n_a), ("n(y)", b.size),
            ("Mean(x)", float(a.mean())), ("Mean(y)", float(b.mean())),
        ],
        warnings=[] if n_perm >= 1000 else
                 [f"only {n_perm} permutations - p-value may be discrete; "
                  "consider increasing for borderline cases."],
    )
