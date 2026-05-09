# moirais.fn — function file (hadesllm/moirais)
"""Maximum-likelihood Poisson fit with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def mlepoi(counts: Union[Sequence, np.ndarray]):
    """MLE Poisson fit: lambda_hat = mean."""
    from ._richresult import RichResult
    a = np.asarray(counts, dtype=float)
    if np.any(a < 0):
        raise ValueError("counts must be non-negative.")
    lam = float(a.mean())
    var_obs = float(a.var(ddof=1)) if a.size > 1 else float("nan")
    overdisp = var_obs / lam if lam > 0 else float("nan")
    log_lik = float(np.sum(a * np.log(lam) - lam - np.log(np.array(
        [np.math.factorial(int(x)) if x < 21 else 0 for x in a],
        dtype=float)))) if lam > 0 and np.all(a == np.round(a)) and a.max() < 21 else float("nan")
    warnings = []
    if not np.isnan(overdisp) and overdisp > 1.5:
        warnings.append(f"overdispersion: var/mean = {overdisp:.2f} > 1.5; "
                        "Poisson assumes var=mean. Consider negative binomial GLM.")
    if np.any(a != np.round(a)):
        warnings.append("non-integer values in counts; Poisson is for non-negative integers.")
    return RichResult(
        title="Maximum-likelihood Poisson fit",
        summary_lines=[
            ("lambda (rate)", lam),
            ("Sample variance", var_obs),
            ("Variance / mean ratio", overdisp),
            ("n", int(a.size)),
            ("Min observed", int(a.min())), ("Max observed", int(a.max())),
        ],
        warnings=warnings,
        payload={"lambda": lam, "value": lam, "statistic": lam,
                 "variance": var_obs, "overdispersion": overdisp},
    )
