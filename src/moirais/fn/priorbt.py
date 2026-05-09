# moirais.fn — function file (hadesllm/moirais)
"""Beta-Binomial conjugate update with R-style verbose result."""


def priorbt(alpha_prior: float, beta_prior: float,
            successes: int, trials: int):
    """Beta-Binomial conjugate posterior update."""
    from ._richresult import RichResult
    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError(f"prior alpha/beta must be positive; got {alpha_prior}, {beta_prior}.")
    if successes < 0 or trials < successes:
        raise ValueError(f"require 0 <= successes <= trials; got {successes}/{trials}.")
    a = alpha_prior + successes
    b = beta_prior + (trials - successes)
    post_mean = a / (a + b)
    post_mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else float("nan")
    post_var = (a * b) / (((a + b) ** 2) * (a + b + 1))
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    return RichResult(
        title="Beta-Binomial conjugate update",
        summary_lines=[
            ("Posterior alpha", a), ("Posterior beta", b),
            ("Posterior mean", post_mean),
            ("Posterior mode", post_mode),
            ("Posterior variance", post_var),
            ("Prior mean", prior_mean),
            ("Observed proportion", successes / trials if trials > 0 else float("nan")),
            ("Successes / Trials", f"{successes}/{trials}"),
        ],
        interpretation=(f"Beta({alpha_prior},{beta_prior}) prior + {successes}/{trials} "
                        f"-> Beta({a},{b}) posterior; mean shifted from "
                        f"{prior_mean:.3f} to {post_mean:.3f}."),
        payload={"alpha_post": a, "beta_post": b, "post_mean": post_mean,
                 "post_mode": post_mode, "post_var": post_var,
                 "value": post_mean, "statistic": post_mean},
    )
