# moirais.fn — function file (hadesllm/moirais)
"""Maximum-likelihood Normal fit with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def mlenrm(x: Union[Sequence, np.ndarray]):
    """MLE Normal fit: mu_hat, sigma_hat (1/n divisor)."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        raise ValueError(f"need at least 2 observations, got {a.size}.")
    mu = float(a.mean())
    sigma_mle = float(((a - mu) ** 2).mean() ** 0.5)
    sigma_unbiased = float(a.std(ddof=1))
    log_lik = float(-0.5 * a.size * np.log(2 * np.pi * sigma_mle ** 2)
                    - 0.5 * np.sum((a - mu) ** 2) / sigma_mle ** 2) if sigma_mle > 0 else float("-inf")
    return RichResult(
        title="Maximum-likelihood Normal fit",
        summary_lines=[
            ("mu (mean)", mu),
            ("sigma (MLE, 1/n divisor)", sigma_mle),
            ("sigma (unbiased, 1/(n-1))", sigma_unbiased),
            ("Log-likelihood at MLE", log_lik),
            ("n", int(a.size)),
        ],
        warnings=[] if a.size >= 30 else
                 [f"small sample n={a.size}; MLE sigma is downward-biased - "
                  "consider sigma_unbiased for inference."],
        payload={"mu": mu, "sigma": sigma_mle,
                 "sigma_unbiased": sigma_unbiased, "loglik": log_lik},
    )
