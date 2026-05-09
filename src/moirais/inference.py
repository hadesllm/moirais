"""
Inference module
Wraps frequentist tests and bayesian inference algorithms.

This module provides:

1. R-compatible distribution functions (d/p/q/r prefix pattern) for continuous
   and discrete univariate distributions, all backed by scipy.stats.
2. Hypothesis testing: t-tests, chi-square, Fisher exact, ANOVA, nonparametric tests,
   normality tests, and variance-equality tests.
3. Confidence intervals for proportions, rates, ratios, and differences.
4. Effect-size estimators: Cohen's d, Hedges' g, eta-squared, omega-squared,
   Cramér's V, phi coefficient, point-biserial r, Kendall's tau, Spearman's rho.
5. Power analysis: t-test, proportion test, ANOVA, and logistic-regression sample
   size, all using statsmodels power classes.

References
----------
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    Lawrence Erlbaum Associates.
Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
    Chapman & Hall/CRC.
Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley.
Newcombe, R. G. (1998). Interval estimation for the difference between independent
    proportions. Statistics in Medicine, 17, 873-890.
Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
    inference. Journal of the American Statistical Association, 22, 209-212.
"""

import math
from collections.abc import Callable
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.power import (
    FTestAnovaPower,
    NormalIndPower,
    TTestIndPower,
    TTestPower,
)


def calculate_interaction_power(sample_size: int, alpha: float = 0.05, effect_size: float = 0.2) -> float:
    """
    Calculate statistical power for an ANOVA F-test, used as an approximation
    for interaction-term power in the CPADS study design.

    The formula ``1 - exp(-(n * f) / 50)`` that appeared in earlier versions
    of this function is **not a valid power formula** — it is a heuristic
    approximation with no grounding in probability theory and produces values
    that systematically diverge from the correct power at moderate sample
    sizes.  It has been replaced by the exact normal-approximation F-test
    power computation from :mod:`statsmodels.stats.power`.

    The implemented estimator solves:

    .. math::

        \\text{power} = 1 - F_{df_1, df_2, \\lambda}(F_{\\text{crit}})

    where :math:`F_{\\text{crit}}` is the critical value under the central
    F-distribution at level ``alpha``, :math:`\\lambda = n \\cdot f^2` is the
    non-centrality parameter, :math:`f` is Cohen's f effect size, :math:`df_1`
    defaults to 1 (single-df test), and :math:`df_2 = n - df_1 - 1`.

    :param sample_size: Total number of observations.
    :type sample_size: int
    :param alpha: Type I error probability limit (significance level), defaults to 0.05.
    :type alpha: float, optional
    :param effect_size: Cohen's *f* effect size for the interaction term,
        defaults to 0.2 (conventionally "small").
    :type effect_size: float, optional
    :return: The estimated statistical power (1 - beta), clipped to [0, 1].
    :rtype: float

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
    (2nd ed.). Lawrence Erlbaum Associates.

    Faul, F., Erdfelder, E., Lang, A.-G., & Buchner, A. (2007). G*Power 3: A
    flexible statistical power analysis program for the social, behavioral, and
    biomedical sciences. *Behavior Research Methods*, 39(2), 175–191.
    https://doi.org/10.3758/BF03193146
    """
    analysis = FTestAnovaPower()
    # solve_power returns a float; clip defensively to [0, 1].
    power = analysis.solve_power(
        effect_size=float(effect_size),
        nobs=float(sample_size),
        alpha=float(alpha),
    )
    return float(np.clip(power, 0.0, 1.0))


def bootstrap_ci(
    estimation_func: Callable,
    data: pd.DataFrame,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute non-parametric percentile bootstrap confidence intervals.

    A fixed random seed is used so that results are **reproducible** across
    calls with the same inputs.  The default seed (42) is arbitrary; pass a
    different ``seed`` value if you need independent bootstrap draws in a
    simulation study.

    The percentile bootstrap is used rather than the BCa or studentised
    bootstrap because it requires only a point-estimate function and makes
    no symmetry assumptions.  For highly skewed sampling distributions,
    consider the BCa bootstrap (Efron & Tibshirani, 1993, Chapter 14).

    :param estimation_func: A callable that takes a DataFrame sample and
        returns a numeric point estimate.
    :type estimation_func: Callable
    :param data: The pandas DataFrame representing the sample population.
    :type data: pandas.DataFrame
    :param n_iterations: Number of bootstrap iterations to perform,
        defaults to 1000.
    :type n_iterations: int, optional
    :param alpha: The significance level for the interval (e.g. 0.05 for a
        95% CI), defaults to 0.05.
    :type alpha: float, optional
    :param seed: Random seed for reproducibility.  Default 42.
    :type seed: int, optional
    :return: A tuple ``(lower, upper)`` containing the percentile bootstrap
        confidence interval bounds.
    :rtype: tuple[float, float]

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593
    """
    # Set the random seed BEFORE any stochastic operation so that all
    # bootstrap draws are deterministic given the same (data, seed) pair.
    np.random.seed(seed)

    estimates = []
    n = len(data)
    for _ in range(n_iterations):
        sample = data.sample(n=n, replace=True)
        estimates.append(estimation_func(sample))

    lower = float(np.percentile(estimates, (alpha / 2) * 100))
    upper = float(np.percentile(estimates, (1 - alpha / 2) * 100))
    return lower, upper


# ===========================================================================
# SECTION 1 — R-COMPATIBLE DISTRIBUTION FUNCTIONS
# ===========================================================================
# Each distribution follows the R naming convention:
#   d<name>(x, ...)  — probability density / mass function
#   p<name>(x, ...)  — cumulative distribution function
#   q<name>(p, ...)  — quantile (inverse CDF)
#   r<name>(n, ...)  — random sample generation
#
# The ``log`` flag (where present) returns log-probability for numerical
# stability in likelihood computations; it is equivalent to applying
# np.log() to the non-logged value but avoids intermediate underflow.
#
# All functions delegate to scipy.stats equivalents, so numerical accuracy
# matches the Cephes / Boost math library underlying scipy.
# ===========================================================================

# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------


def dnorm(
    x: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Normal distribution probability density function.

    Computes the density of :math:`X \\sim \\mathcal{N}(\\mu, \\sigma^2)`:

    .. math::

        f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}
               \\exp\\!\\left(-\\frac{(x - \\mu)^2}{2\\sigma^2}\\right)

    Mirrors R's ``dnorm(x, mean, sd, log)``.

    :param x: Quantile(s) at which to evaluate the density.
    :param mean: Mean of the distribution (mu). Default 0.0.
    :param sd: Standard deviation (sigma > 0). Default 1.0.
    :param log: If True return log-density (for numerical stability). Default False.
    :return: Density value(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). dnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    dist = stats.norm(loc=mean, scale=sd)
    result = dist.logpdf(x) if log else dist.pdf(x)
    return result


def pnorm(
    x: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Normal distribution cumulative distribution function.

    Mirrors R's ``pnorm(x, mean, sd, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param lower_tail: If True (default) compute P(X <= x); else P(X > x).
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). pnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    dist = stats.norm(loc=mean, scale=sd)
    if lower_tail:
        # logcdf = log(P(X <= x)) — correct for lower tail
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        # logsf = log(P(X > x)) = log(1 - CDF) — correct for upper tail
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qnorm(
    p: Union[float, np.ndarray], mean: float = 0.0, sd: float = 1.0, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Normal distribution quantile function (inverse CDF).

    Mirrors R's ``qnorm(p, mean, sd, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1); or log-probabilities if log=True.
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param lower_tail: If True (default), p = P(X <= x); else p = P(X > x).
    :param log: If True, p is treated as log(probability). Default False.
    :return: Quantile(s).
    :raises ValueError: If sd <= 0.

    References
    ----------
    R Core Team (2024). qnorm {stats}. R documentation.
    """
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.norm(loc=mean, scale=sd).ppf(p_arr)


def rnorm(n: int, mean: float = 0.0, sd: float = 1.0, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a normal distribution.

    Mirrors R's ``rnorm(n, mean, sd)``.

    :param n: Number of observations to draw (> 0).
    :param mean: Mean. Default 0.0.
    :param sd: Standard deviation (> 0). Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or sd <= 0.

    References
    ----------
    R Core Team (2024). rnorm {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=sd, size=n)


# ---------------------------------------------------------------------------
# Student's t-distribution
# ---------------------------------------------------------------------------


def dt(
    x: Union[float, np.ndarray], df: float, ncp: float | None = None, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Student's t-distribution probability density function.

    Mirrors R's ``dt(x, df, ncp, log)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param ncp: Non-centrality parameter. If None, central t is used.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). dt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.nct(df=df, nc=ncp) if ncp is not None else stats.t(df=df)
    return dist.logpdf(x) if log else dist.pdf(x)


def pt(x: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False) -> Union[float, np.ndarray]:
    """
    Student's t-distribution CDF.

    Mirrors R's ``pt(x, df, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True compute P(T <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). pt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.t(df=df)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qt(p: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False) -> Union[float, np.ndarray]:
    """
    Student's t-distribution quantile function.

    Mirrors R's ``qt(p, df, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True p = P(T <= x). Default True.
    :param log: If True, p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). qt {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.t(df=df).ppf(p_arr)


# ---------------------------------------------------------------------------
# Chi-squared distribution
# ---------------------------------------------------------------------------


def dchisq(x: Union[float, np.ndarray], df: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution PDF.

    Mirrors R's ``dchisq(x, df, log)``.

    :param x: Quantile(s) (>= 0).
    :param df: Degrees of freedom (> 0).
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). dchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.chi2(df=df)
    return dist.logpdf(x) if log else dist.pdf(x)


def pchisq(
    x: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution CDF.

    Mirrors R's ``pchisq(x, df, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). pchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    dist = stats.chi2(df=df)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qchisq(
    p: Union[float, np.ndarray], df: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Chi-squared distribution quantile function.

    Mirrors R's ``qchisq(p, df, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param df: Degrees of freedom (> 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If df <= 0.

    References
    ----------
    R Core Team (2024). qchisq {stats}. R documentation.
    """
    if df <= 0:
        raise ValueError(f"df must be > 0, got {df}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.chi2(df=df).ppf(p_arr)


# ---------------------------------------------------------------------------
# Binomial distribution
# ---------------------------------------------------------------------------


def dbinom(x: Union[int, np.ndarray], size: int, prob: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Binomial distribution probability mass function.

    Computes :math:`P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}`.

    Mirrors R's ``dbinom(x, size, prob, log)``.

    :param x: Number of successes (integer >= 0).
    :param size: Number of trials (n >= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param log: If True return log-PMF. Default False.
    :return: PMF value(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). dbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    dist = stats.binom(n=size, p=prob)
    return dist.logpmf(x) if log else dist.pmf(x)


def pbinom(
    x: Union[int, np.ndarray], size: int, prob: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Binomial distribution CDF.

    Mirrors R's ``pbinom(x, size, prob, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param size: Number of trials (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). pbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    dist = stats.binom(n=size, p=prob)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qbinom(
    p: Union[float, np.ndarray], size: int, prob: float, lower_tail: bool = True, log: bool = False
) -> Union[int, np.ndarray]:
    """
    Binomial distribution quantile function.

    Returns the smallest integer k such that P(X <= k) >= p.
    Mirrors R's ``qbinom(p, size, prob, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param size: Number of trials (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s) as integer(s).
    :raises ValueError: If size < 0 or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). qbinom {stats}. R documentation.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.binom(n=size, p=prob).ppf(p_arr)


def rbinom(n: int, size: int, prob: float, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a binomial distribution.

    Mirrors R's ``rbinom(n, size, prob)``.

    :param n: Number of observations to draw (> 0).
    :param size: Number of trials per observation (>= 0).
    :param prob: Success probability per trial (in [0, 1]).
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D integer array of length n.
    :raises ValueError: If n <= 0, size < 0, or prob not in [0, 1].

    References
    ----------
    R Core Team (2024). rbinom {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}.")
    rng = np.random.default_rng(seed)
    return rng.binomial(n=size, p=prob, size=n)


# ---------------------------------------------------------------------------
# Poisson distribution
# ---------------------------------------------------------------------------


def dpois(x: Union[int, np.ndarray], lambda_: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Poisson distribution probability mass function.

    Computes :math:`P(X = k) = e^{-\\lambda} \\lambda^k / k!`.

    Mirrors R's ``dpois(x, lambda, log)``.

    :param x: Non-negative integer value(s).
    :param lambda_: Rate parameter (lambda > 0).
    :param log: If True return log-PMF. Default False.
    :return: PMF value(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). dpois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    dist = stats.poisson(mu=lambda_)
    return dist.logpmf(x) if log else dist.pmf(x)


def ppois(
    x: Union[int, np.ndarray], lambda_: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Poisson distribution CDF.

    Mirrors R's ``ppois(x, lambda, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param lambda_: Rate parameter (lambda > 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). ppois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    dist = stats.poisson(mu=lambda_)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qpois(
    p: Union[float, np.ndarray], lambda_: float, lower_tail: bool = True, log: bool = False
) -> Union[int, np.ndarray]:
    """
    Poisson distribution quantile function.

    Mirrors R's ``qpois(p, lambda, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param lambda_: Rate parameter (lambda > 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If lambda_ <= 0.

    References
    ----------
    R Core Team (2024). qpois {stats}. R documentation.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.poisson(mu=lambda_).ppf(p_arr)


def rpois(n: int, lambda_: float, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a Poisson distribution.

    Mirrors R's ``rpois(n, lambda)``.

    :param n: Number of observations to draw (> 0).
    :param lambda_: Rate parameter (lambda > 0).
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D integer array of length n.
    :raises ValueError: If n <= 0 or lambda_ <= 0.

    References
    ----------
    R Core Team (2024). rpois {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be > 0, got {lambda_}.")
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lambda_, size=n)


# ---------------------------------------------------------------------------
# Beta distribution
# ---------------------------------------------------------------------------


def dbeta(x: Union[float, np.ndarray], alpha: float, beta: float, log: bool = False) -> Union[float, np.ndarray]:
    """
    Beta distribution probability density function.

    Mirrors R's ``dbeta(x, shape1, shape2, log)``.

    :param x: Value(s) in [0, 1].
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). dbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    dist = stats.beta(a=alpha, b=beta)
    return dist.logpdf(x) if log else dist.pdf(x)


def pbeta(
    x: Union[float, np.ndarray], alpha: float, beta: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Beta distribution CDF.

    Mirrors R's ``pbeta(x, shape1, shape2, lower.tail, log.p)``.

    :param x: Value(s) in [0, 1].
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). pbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    dist = stats.beta(a=alpha, b=beta)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def qbeta(
    p: Union[float, np.ndarray], alpha: float, beta: float, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Beta distribution quantile function.

    Mirrors R's ``qbeta(p, shape1, shape2, lower.tail, log.p)``.

    :param p: Probability value(s) in (0, 1).
    :param alpha: First shape parameter (> 0).
    :param beta: Second shape parameter (> 0).
    :param lower_tail: If True p = P(X <= x). Default True.
    :param log: If True p is log-probability. Default False.
    :return: Quantile(s).
    :raises ValueError: If alpha <= 0 or beta <= 0.

    References
    ----------
    R Core Team (2024). qbeta {stats}. R documentation.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}.")
    p_arr = np.asarray(p, dtype=float)
    if log:
        p_arr = np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    return stats.beta(a=alpha, b=beta).ppf(p_arr)


# ---------------------------------------------------------------------------
# Gamma distribution
# ---------------------------------------------------------------------------


def dgamma(
    x: Union[float, np.ndarray], shape: float, rate: float = 1.0, scale: float | None = None, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Gamma distribution probability density function.

    Supports both rate and scale parameterisation (rate = 1/scale).
    When ``scale`` is provided it takes precedence over ``rate``.
    Mirrors R's ``dgamma(x, shape, rate, scale, log)``.

    :param x: Quantile(s) (>= 0).
    :param shape: Shape parameter alpha (> 0).
    :param rate: Rate parameter beta = 1/scale (> 0). Default 1.0.
    :param scale: Scale parameter theta = 1/rate. Overrides rate if provided.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If shape <= 0, or both rate and scale are inconsistent.

    References
    ----------
    R Core Team (2024). dgamma {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    effective_scale = 1.0 / rate if scale is None else scale
    if effective_scale <= 0:
        raise ValueError(f"Effective scale must be > 0, got {effective_scale}.")
    dist = stats.gamma(a=shape, scale=effective_scale)
    return dist.logpdf(x) if log else dist.pdf(x)


def pgamma(
    x: Union[float, np.ndarray],
    shape: float,
    rate: float = 1.0,
    scale: float | None = None,
    lower_tail: bool = True,
    log: bool = False,
) -> Union[float, np.ndarray]:
    """
    Gamma distribution CDF.

    Mirrors R's ``pgamma(x, shape, rate, scale, lower.tail, log.p)``.

    :param x: Quantile(s).
    :param shape: Shape parameter (> 0).
    :param rate: Rate parameter (> 0). Default 1.0.
    :param scale: Scale parameter (overrides rate if provided).
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If shape <= 0.

    References
    ----------
    R Core Team (2024). pgamma {stats}. R documentation.
    """
    if shape <= 0:
        raise ValueError(f"shape must be > 0, got {shape}.")
    effective_scale = 1.0 / rate if scale is None else scale
    if effective_scale <= 0:
        raise ValueError(f"Effective scale must be > 0, got {effective_scale}.")
    dist = stats.gamma(a=shape, scale=effective_scale)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


# ---------------------------------------------------------------------------
# Uniform distribution
# ---------------------------------------------------------------------------


def dunif(
    x: Union[float, np.ndarray], min: float = 0.0, max: float = 1.0, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Uniform distribution probability density function.

    Density is 1/(max - min) for x in [min, max], 0 otherwise.
    Mirrors R's ``dunif(x, min, max, log)``.

    :param x: Value(s).
    :param min: Lower bound of the support. Default 0.0.
    :param max: Upper bound of the support. Default 1.0.
    :param log: If True return log-density. Default False.
    :return: Density value(s).
    :raises ValueError: If min >= max.

    References
    ----------
    R Core Team (2024). dunif {stats}. R documentation.
    """
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    dist = stats.uniform(loc=min, scale=max - min)
    return dist.logpdf(x) if log else dist.pdf(x)


def punif(
    x: Union[float, np.ndarray], min: float = 0.0, max: float = 1.0, lower_tail: bool = True, log: bool = False
) -> Union[float, np.ndarray]:
    """
    Uniform distribution CDF.

    Mirrors R's ``punif(x, min, max, lower.tail, log.p)``.

    :param x: Value(s).
    :param min: Lower bound. Default 0.0.
    :param max: Upper bound. Default 1.0.
    :param lower_tail: If True compute P(X <= x). Default True.
    :param log: If True return log-probability. Default False.
    :return: CDF value(s).
    :raises ValueError: If min >= max.

    References
    ----------
    R Core Team (2024). punif {stats}. R documentation.
    """
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    dist = stats.uniform(loc=min, scale=max - min)
    if lower_tail:
        result = dist.logcdf(x) if log else dist.cdf(x)
    else:
        result = dist.logsf(x) if log else dist.sf(x)
    return result


def runif(n: int, min: float = 0.0, max: float = 1.0, seed: int | None = None) -> np.ndarray:
    """
    Draw a random sample from a uniform distribution.

    Mirrors R's ``runif(n, min, max)``.

    :param n: Number of observations to draw (> 0).
    :param min: Lower bound. Default 0.0.
    :param max: Upper bound. Default 1.0.
    :param seed: Random seed for reproducibility. Default None.
    :return: 1-D array of length n.
    :raises ValueError: If n <= 0 or min >= max.

    References
    ----------
    R Core Team (2024). runif {stats}. R documentation.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}.")
    if min >= max:
        raise ValueError(f"min must be < max, got min={min}, max={max}.")
    rng = np.random.default_rng(seed)
    return rng.uniform(low=min, high=max, size=n)


# ===========================================================================
# SECTION 2 — HYPOTHESIS TESTING
# ===========================================================================


def two_sample_t_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> dict:
    """
    Two-sample t-test (Welch or Student).

    Default is Welch's t-test (``equal_var=False``), which does not assume
    equal population variances and is generally safer (Ruxton, 2006).

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (1-D array-like).
    :param equal_var: If True use Student's t-test; if False (default) use Welch's.
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``ci_diff_lower``,
        ``ci_diff_upper``, ``mean_diff``, ``method``.
    :raises ValueError: If x1 or x2 is empty or alternative is invalid.

    References
    ----------
    Welch, B. L. (1947). The generalization of Student's problem when several
        different population variances are involved. Biometrika, 34(1-2), 28-35.
    Ruxton, G. D. (2006). The unequal variance t-test is an underused alternative
        to Student's t-test and the Mann-Whitney U test. Behavioral Ecology, 17(4).
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")

    t_stat, p_val = stats.ttest_ind(a1, a2, equal_var=equal_var, alternative=alternative)
    mean_diff = float(np.mean(a1) - np.mean(a2))
    # Degrees of freedom: Welch-Satterthwaite formula for unequal-var case
    if equal_var:
        df = float(len(a1) + len(a2) - 2)
    else:
        s1, s2 = np.var(a1, ddof=1), np.var(a2, ddof=1)
        n1, n2 = len(a1), len(a2)
        num = (s1 / n1 + s2 / n2) ** 2
        denom = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        df = float(num / denom) if denom > 0 else float(n1 + n2 - 2)
    # Two-sided CI for the mean difference
    se_diff = float(np.sqrt(np.var(a1, ddof=1) / len(a1) + np.var(a2, ddof=1) / len(a2)))
    t_crit = float(stats.t(df=df).ppf(0.975))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "t": float(t_stat),
        "df": df,
        "p_value": float(p_val),
        "mean_diff": mean_diff,
        "ci_diff_lower": ci_lower,
        "ci_diff_upper": ci_upper,
        "method": "Welch two-sample t-test" if not equal_var else "Student two-sample t-test",
    }


def one_sample_t_test(
    x: Union[list, np.ndarray],
    mu0: float = 0.0,
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    One-sample t-test against a specified null mean.

    Tests H0: mu = mu0 vs Ha depending on ``alternative``.

    :param x: Sample data (1-D array-like).
    :param mu0: Null hypothesis mean. Default 0.0.
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``mean``, ``se``,
        ``ci_lower``, ``ci_upper``.
    :raises ValueError: If x has fewer than 2 observations.

    References
    ----------
    Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 2:
        raise ValueError("x must have at least 2 observations.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    t_stat, p_val = stats.ttest_1samp(arr, popmean=mu0, alternative=alternative)
    n = len(arr)
    mean_val = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t(df=n - 1).ppf(0.975))
    return {
        "t": float(t_stat),
        "df": float(n - 1),
        "p_value": float(p_val),
        "mean": mean_val,
        "se": se,
        "ci_lower": mean_val - t_crit * se,
        "ci_upper": mean_val + t_crit * se,
        "method": "One-sample t-test",
    }


def paired_t_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Paired samples t-test.

    Tests H0: mean(x1 - x2) = 0. Requires equal-length samples.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (same length as x1).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``t``, ``df``, ``p_value``, ``mean_diff``,
        ``se_diff``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If samples have different lengths or fewer than 2 pairs.

    References
    ----------
    Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) != len(a2):
        raise ValueError("x1 and x2 must have the same length for a paired test.")
    if len(a1) < 2:
        raise ValueError("At least 2 pairs are required.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    diff = a1 - a2
    t_stat, p_val = stats.ttest_rel(a1, a2, alternative=alternative)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se_diff = float(np.std(diff, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t(df=n - 1).ppf(0.975))
    return {
        "t": float(t_stat),
        "df": float(n - 1),
        "p_value": float(p_val),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": mean_diff - t_crit * se_diff,
        "ci_upper": mean_diff + t_crit * se_diff,
        "method": "Paired t-test",
    }


def chi_square_test(
    observed: Union[list, np.ndarray],
    *,
    expected: Union[list, np.ndarray] | None = None,
) -> dict:
    """
    Chi-square goodness-of-fit (or independence) test.

    For a 1-D ``observed`` array, performs goodness-of-fit against ``expected``
    (uniform if None). For a 2-D array, performs a chi-square test of
    independence (``expected`` is ignored).

    :param observed: Observed frequencies. 1-D or 2-D array-like.
    :param expected: Expected frequencies for 1-D case. Must sum to same total
        as observed. Uniform if None.
    :return: dict with keys ``chi2``, ``df``, ``p_value``, ``expected``.
    :raises ValueError: If observed contains negative values.

    References
    ----------
    Agresti, A. (2013). Categorical Data Analysis (3rd ed.). Wiley. (Chapter 1.)
    """
    obs = np.asarray(observed, dtype=float)
    if np.any(obs < 0):
        raise ValueError("Observed frequencies must be non-negative.")
    if obs.ndim == 2:
        chi2_stat, p_val, df, exp = stats.chi2_contingency(obs)
        return {
            "chi2": float(chi2_stat),
            "df": int(df),
            "p_value": float(p_val),
            "expected": exp,
            "method": "Chi-square test of independence",
        }
    # 1-D goodness-of-fit
    exp_arr = np.ones_like(obs) * obs.sum() / len(obs) if expected is None else np.asarray(expected, dtype=float)
    chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp_arr)
    df = len(obs) - 1
    return {
        "chi2": float(chi2_stat),
        "df": df,
        "p_value": float(p_val),
        "expected": exp_arr,
        "method": "Chi-square goodness-of-fit test",
    }


def fisher_exact_test(table_2x2: Union[list, np.ndarray]) -> dict:
    """
    Fisher's exact test for a 2x2 contingency table.

    Appropriate when expected cell counts are small (< 5) and the chi-square
    approximation is unreliable.

    :param table_2x2: A 2x2 array-like [[a, b], [c, d]].
    :return: dict with keys ``odds_ratio``, ``p_value``.
    :raises ValueError: If the table is not 2x2 or contains negative values.

    References
    ----------
    Fisher, R. A. (1935). The logic of inductive inference. Journal of the Royal
        Statistical Society, 98(1), 39-82.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must have shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Contingency table entries must be non-negative.")
    odds_ratio, p_val = stats.fisher_exact(tbl.astype(int))
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_val),
        "method": "Fisher's exact test",
    }


def anova_one_way(*groups) -> dict:
    """
    One-way ANOVA F-test.

    Tests H0: mu_1 = mu_2 = ... = mu_k. Assumes independent observations,
    normality within each group, and homoscedasticity.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``F``, ``df_between``, ``df_within``, ``p_value``,
        ``eta_squared``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Fisher, R. A. (1925). Statistical Methods for Research Workers. Oliver & Boyd.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if len(groups) < 2:
        raise ValueError("anova_one_way requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    for i, a in enumerate(arrays):
        if len(a) < 1:
            raise ValueError(f"Group {i} is empty.")
    f_stat, p_val = stats.f_oneway(*arrays)
    k = len(arrays)
    N = sum(len(a) for a in arrays)
    df_between = k - 1
    df_within = N - k
    # Eta-squared: SS_between / SS_total
    grand_mean = np.mean(np.concatenate(arrays))
    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in arrays)
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in arrays)
    ss_total = ss_between + ss_within
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return {
        "F": float(f_stat),
        "df_between": df_between,
        "df_within": df_within,
        "p_value": float(p_val),
        "eta_squared": eta_sq,
        "method": "One-way ANOVA",
    }


def kruskal_wallis_test(*groups) -> dict:
    """
    Kruskal-Wallis H-test (non-parametric one-way ANOVA).

    A rank-based alternative to one-way ANOVA that does not assume normality.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``H``, ``df``, ``p_value``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion
        variance analysis. Journal of the American Statistical Association, 47.
    """
    if len(groups) < 2:
        raise ValueError("kruskal_wallis_test requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    h_stat, p_val = stats.kruskal(*arrays)
    return {
        "H": float(h_stat),
        "df": len(groups) - 1,
        "p_value": float(p_val),
        "method": "Kruskal-Wallis H-test",
    }


def mann_whitney_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Non-parametric alternative to the two-sample t-test for ordinal or
    non-normally distributed data.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (1-D array-like).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``U``, ``p_value``.
    :raises ValueError: If x1 or x2 is empty or alternative is invalid.

    References
    ----------
    Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random
        variables is stochastically larger than the other. Annals of Mathematical
        Statistics, 18(1), 50-60.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 1:
        raise ValueError("x1 must not be empty.")
    if len(a2) < 1:
        raise ValueError("x2 must not be empty.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    u_stat, p_val = stats.mannwhitneyu(a1, a2, alternative=alternative)
    return {
        "U": float(u_stat),
        "p_value": float(p_val),
        "method": "Mann-Whitney U test",
    }


def wilcoxon_signed_rank_test(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    alternative: str = "two-sided",
) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to the paired t-test.

    :param x1: First sample (1-D array-like).
    :param x2: Second sample (same length as x1).
    :param alternative: ``"two-sided"``, ``"greater"``, or ``"less"``.
    :return: dict with keys ``statistic``, ``p_value``.
    :raises ValueError: If samples have different lengths or fewer than 2 pairs.

    References
    ----------
    Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics
        Bulletin, 1(6), 80-83.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) != len(a2):
        raise ValueError("x1 and x2 must have the same length.")
    if len(a1) < 2:
        raise ValueError("At least 2 pairs are required.")
    valid_alternatives = {"two-sided", "greater", "less"}
    if alternative not in valid_alternatives:
        raise ValueError(f"alternative must be one of {valid_alternatives}.")
    w_stat, p_val = stats.wilcoxon(a1, a2, alternative=alternative)
    return {
        "statistic": float(w_stat),
        "p_value": float(p_val),
        "method": "Wilcoxon signed-rank test",
    }


def shapiro_wilk_test(x: Union[list, np.ndarray]) -> dict:
    """
    Shapiro-Wilk test for normality.

    Tests H0: the sample comes from a normal distribution.
    Recommended for n <= 2000; use Kolmogorov-Smirnov for larger samples.

    :param x: Sample data (1-D array-like, 3 <= n <= 5000).
    :return: dict with keys ``W``, ``p_value``, ``is_normal``
        (True if p_value > 0.05).
    :raises ValueError: If x has fewer than 3 observations.

    References
    ----------
    Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for
        normality. Biometrika, 52(3-4), 591-611.
    """
    arr = np.asarray(x, dtype=float)
    if len(arr) < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 observations.")
    w_stat, p_val = stats.shapiro(arr)
    return {
        "W": float(w_stat),
        "p_value": float(p_val),
        "is_normal": float(p_val) > 0.05,
        "method": "Shapiro-Wilk test",
    }


def levene_test(*groups) -> dict:
    """
    Levene's test for equality of variances.

    Tests H0: sigma_1^2 = sigma_2^2 = ... = sigma_k^2.
    Uses the median (Brown-Forsythe variant) for robustness to non-normality.

    :param groups: Two or more 1-D array-like group samples.
    :return: dict with keys ``W``, ``df_between``, ``df_within``, ``p_value``.
    :raises ValueError: If fewer than 2 groups are provided.

    References
    ----------
    Levene, H. (1960). Robust tests for equality of variances. In I. Olkin (Ed.),
        Contributions to Probability and Statistics. Stanford University Press.
    Brown, M. B., & Forsythe, A. B. (1974). Robust tests for equality of
        variances. Journal of the American Statistical Association, 69, 364-367.
    """
    if len(groups) < 2:
        raise ValueError("levene_test requires at least 2 groups.")
    arrays = [np.asarray(g, dtype=float) for g in groups]
    w_stat, p_val = stats.levene(*arrays, center="median")
    k = len(arrays)
    N = sum(len(a) for a in arrays)
    return {
        "W": float(w_stat),
        "df_between": k - 1,
        "df_within": N - k,
        "p_value": float(p_val),
        "method": "Levene test (Brown-Forsythe variant)",
    }


# ===========================================================================
# SECTION 3 — CONFIDENCE INTERVALS
# ===========================================================================


def proportion_ci(
    successes: int,
    n: int,
    *,
    method: str = "wilson",
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Confidence interval for a single proportion.

    Three methods are supported:
    - ``"wilson"`` (default): Wilson score interval — recommended for small n
      and proportions near 0 or 1 (Brown, Cai & DasGupta, 2001).
    - ``"clopper-pearson"``: Exact Clopper-Pearson interval — conservative.
    - ``"agresti-coull"``: Agresti-Coull interval — good coverage for moderate n.

    :param successes: Number of successes (0 <= successes <= n).
    :param n: Total number of trials (> 0).
    :param method: CI method. Default ``"wilson"``.
    :param alpha: Significance level (default 0.05 → 95% CI).
    :return: Tuple ``(lower, upper)``.
    :raises ValueError: If successes < 0, n <= 0, or alpha not in (0, 1).

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
        statistical inference. JASA, 22, 209-212.
    Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
        limits illustrated in the case of the binomial. Biometrika, 26, 404-413.
    Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
        binomial proportion. Statistical Science, 16(2), 101-133.
    """
    if successes < 0:
        raise ValueError("successes must be >= 0.")
    if n <= 0:
        raise ValueError("n must be > 0.")
    if successes > n:
        raise ValueError("successes cannot exceed n.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    p_hat = successes / n
    z = float(stats.norm.ppf(1 - alpha / 2))

    if method == "wilson":
        denom = 1 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))
    elif method == "clopper-pearson":
        lower = float(stats.beta.ppf(alpha / 2, successes, n - successes + 1)) if successes > 0 else 0.0
        upper = float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)) if successes < n else 1.0
        return lower, upper
    elif method == "agresti-coull":
        n_tilde = n + z**2
        p_tilde = (successes + z**2 / 2) / n_tilde
        half_width = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        return float(max(0.0, p_tilde - half_width)), float(min(1.0, p_tilde + half_width))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'wilson', 'clopper-pearson', or 'agresti-coull'.")


def rate_ratio_ci(n1: int, t1: float, n2: int, t2: float, *, alpha: float = 0.05) -> dict:
    """
    Incidence rate ratio (IRR) with asymptotic log-normal confidence interval.

    IRR = (n1/t1) / (n2/t2)

    The CI uses the delta method on the log scale:
    SE(log IRR) = sqrt(1/n1 + 1/n2) (assuming Poisson counts).

    :param n1: Event count in group 1 (>= 0).
    :param t1: Person-time in group 1 (> 0).
    :param n2: Event count in group 2 (>= 0).
    :param t2: Person-time in group 2 (> 0).
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``irr``, ``log_irr``, ``se_log_irr``, ``ci_lower``,
        ``ci_upper``, ``p_value``.
    :raises ValueError: If t1 <= 0, t2 <= 0, or n1/n2 < 0.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern Epidemiology
        (3rd ed.). Lippincott Williams & Wilkins. (Chapter 14.)
    """
    if t1 <= 0 or t2 <= 0:
        raise ValueError("Person-time values t1 and t2 must be > 0.")
    if n1 < 0 or n2 < 0:
        raise ValueError("Event counts n1 and n2 must be >= 0.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    r1 = n1 / t1
    r2 = n2 / t2
    irr = r1 / r2 if r2 > 0 else float("inf")
    log_irr = math.log(irr) if irr > 0 else float("-inf")
    # SE of log IRR under Poisson: sqrt(1/n1 + 1/n2)
    se_log = math.sqrt(1.0 / n1 + 1.0 / n2) if (n1 > 0 and n2 > 0) else float("inf")
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = math.exp(log_irr - z * se_log) if math.isfinite(log_irr) else 0.0
    ci_upper = math.exp(log_irr + z * se_log) if math.isfinite(log_irr) else float("inf")
    # Two-sided Wald z-test on log scale
    z_stat = log_irr / se_log if se_log > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "irr": irr,
        "log_irr": log_irr,
        "se_log_irr": se_log,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def odds_ratio_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Odds ratio with exact (Baptista-Pike) confidence interval from scipy.

    For a 2x2 table [[a, b], [c, d]]:
    OR = (a * d) / (b * c)

    :param table_2x2: 2x2 array [[exposed_case, exposed_control],
        [unexposed_case, unexposed_control]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``odds_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``.
    :raises ValueError: If table is not 2x2 or contains negatives.

    References
    ----------
    Baptista, J., & Pike, M. C. (1977). Exact two-sided confidence limits for the
        odds ratio in a 2 × 2 table. Applied Statistics, 26(2), 214-220.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    # Fisher exact gives odds ratio and two-sided p-value
    or_point, p_val = stats.fisher_exact(tbl.astype(int))
    # For CI, use the Woolf log-normal CI (well-defined when all cells > 0)
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    if all(v > 0 for v in [a, b, c, d]):
        log_or = math.log(a * d / (b * c))
        se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
    else:
        # Haldane-Anscombe correction: add 0.5 to all cells
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        log_or = math.log(a * d / (b * c))
        se_log_or = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
    return {
        "odds_ratio": float(or_point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_val),
    }


def risk_ratio_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Risk ratio (relative risk) with log-normal Wald confidence interval.

    For a 2x2 table [[a, b], [c, d]] where rows are exposure and
    columns are outcome:
    RR = [a/(a+b)] / [c/(c+d)]

    :param table_2x2: 2x2 array [[exposed_outcome, exposed_no_outcome],
        [unexposed_outcome, unexposed_no_outcome]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``risk_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``.
    :raises ValueError: If table is not 2x2 or contains negatives.

    References
    ----------
    Rothman, K. J., Greenland, S., & Lash, T. L. (2008). Modern Epidemiology
        (3rd ed.). Lippincott Williams & Wilkins. (Chapter 4.)
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    n1 = a + b  # exposed
    n2 = c + d  # unexposed
    if n1 == 0 or n2 == 0:
        raise ValueError("Row totals must be > 0.")
    p1 = a / n1
    p2 = c / n2
    if p2 == 0:
        return {"risk_ratio": float("inf"), "ci_lower": float("nan"), "ci_upper": float("nan"), "p_value": float("nan")}
    rr = p1 / p2
    log_rr = math.log(rr)
    # SE(log RR) = sqrt((1-p1)/(a) + (1-p2)/(c)) via delta method
    se_log_rr = math.sqrt((1 - p1) / a + (1 - p2) / c) if (a > 0 and c > 0) else float("inf")
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = math.exp(log_rr - z * se_log_rr)
    ci_upper = math.exp(log_rr + z * se_log_rr)
    z_stat = log_rr / se_log_rr if se_log_rr > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "risk_ratio": float(rr),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_val),
    }


def risk_difference_ci(table_2x2: Union[list, np.ndarray], *, alpha: float = 0.05) -> dict:
    """
    Risk difference (absolute risk difference, ARD) with Newcombe's CI.

    RD = p1 - p2 = a/(a+b) - c/(c+d)

    Uses the Newcombe (1998) score-based method, which has better coverage
    than the standard Wald interval, especially near the boundary.

    :param table_2x2: 2x2 array [[a, b], [c, d]].
    :param alpha: Significance level. Default 0.05.
    :return: dict with keys ``risk_difference``, ``ci_lower``, ``ci_upper``,
        ``p_value``.
    :raises ValueError: If table is not 2x2, contains negatives, or row totals are 0.

    References
    ----------
    Newcombe, R. G. (1998). Interval estimation for the difference between
        independent proportions: comparison of eleven methods. Statistics in
        Medicine, 17, 873-890.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    n1 = a + b
    n2 = c + d
    if n1 == 0 or n2 == 0:
        raise ValueError("Row totals must be > 0.")
    p1 = a / n1
    p2 = c / n2
    rd = p1 - p2
    # Newcombe hybrid Wilson score CI for the difference
    z = float(stats.norm.ppf(1 - alpha / 2))

    # Wilson CI for each proportion
    def _wilson_bounds(x_count, n_total):
        denom = 1 + z**2 / n_total
        centre = (x_count / n_total + z**2 / (2 * n_total)) / denom
        half = z * math.sqrt((x_count / n_total) * (1 - x_count / n_total) / n_total + z**2 / (4 * n_total**2)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    l1, u1 = _wilson_bounds(a, n1)
    l2, u2 = _wilson_bounds(c, n2)
    ci_lower = rd - z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_upper = rd + z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    # Newcombe's refined bounds using Wilson limits
    ci_lower_nc = rd - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    ci_upper_nc = rd + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    # Two-sided z-test
    se_rd = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_stat = rd / se_rd if se_rd > 0 else float("nan")
    p_val = 2.0 * float(stats.norm.sf(abs(z_stat))) if math.isfinite(z_stat) else float("nan")
    return {
        "risk_difference": float(rd),
        "ci_lower": float(ci_lower_nc),
        "ci_upper": float(ci_upper_nc),
        "p_value": float(p_val),
    }


# ===========================================================================
# SECTION 4 — EFFECT SIZES
# ===========================================================================


def cohens_d(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
    *,
    pooled: bool = True,
) -> float:
    """
    Cohen's d effect size for two independent groups.

    Using pooled SD (default):
    d = (mean_1 - mean_2) / s_pooled

    where s_pooled = sqrt([(n1-1)*s1^2 + (n2-1)*s2^2] / (n1+n2-2)).

    Using the control-group SD (``pooled=False``):
    d = (mean_1 - mean_2) / s_2

    Conventional benchmarks: |d| = 0.2 small, 0.5 medium, 0.8 large
    (Cohen, 1988).

    :param x1: First group sample.
    :param x2: Second group sample (reference/control when pooled=False).
    :param pooled: If True use pooled SD denominator. Default True.
    :return: Cohen's d (signed float).
    :raises ValueError: If either sample has fewer than 2 observations.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
        (2nd ed.). Lawrence Erlbaum Associates.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    mean_diff = np.mean(a1) - np.mean(a2)
    if pooled:
        n1, n2 = len(a1), len(a2)
        s1_sq = np.var(a1, ddof=1)
        s2_sq = np.var(a2, ddof=1)
        s_pooled = math.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
        return float(mean_diff / s_pooled) if s_pooled > 0 else float("nan")
    else:
        s2 = float(np.std(a2, ddof=1))
        return float(mean_diff / s2) if s2 > 0 else float("nan")


def hedges_g(
    x1: Union[list, np.ndarray],
    x2: Union[list, np.ndarray],
) -> float:
    """
    Hedges' g effect size (bias-corrected Cohen's d).

    Applies Hedges' correction factor J(m) to reduce small-sample bias:
    g = d * J(m),  where m = n1 + n2 - 2
    J(m) ≈ 1 - 3 / (4m - 1)

    :param x1: First group sample.
    :param x2: Second group sample.
    :return: Hedges' g (signed float).
    :raises ValueError: If either sample has fewer than 2 observations.

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
        size and related estimators. Journal of Educational Statistics, 6(2), 107-128.
    """
    a1 = np.asarray(x1, dtype=float)
    a2 = np.asarray(x2, dtype=float)
    if len(a1) < 2:
        raise ValueError("x1 must have at least 2 observations.")
    if len(a2) < 2:
        raise ValueError("x2 must have at least 2 observations.")
    d = cohens_d(a1, a2, pooled=True)
    m = len(a1) + len(a2) - 2
    # Hedges correction factor: exact via gamma functions; approximate for large m
    if m > 0:
        j = math.exp(math.lgamma(m / 2) - math.log(math.sqrt(m / 2)) - math.lgamma((m - 1) / 2))
    else:
        j = 1.0
    return float(d * j) if math.isfinite(d) else float("nan")


def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """
    Eta-squared (eta^2) from ANOVA F-statistic.

    eta^2 = SS_between / SS_total = (df_between * F) / (df_between * F + df_within)

    Conventional benchmarks: 0.01 small, 0.06 medium, 0.14 large (Cohen, 1988).

    :param f_stat: F-statistic from one-way ANOVA (>= 0).
    :param df_between: Between-groups degrees of freedom (k - 1).
    :param df_within: Within-groups degrees of freedom (N - k).
    :return: Eta-squared in [0, 1].
    :raises ValueError: If f_stat < 0 or df values are non-positive.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}.")
    if df_between <= 0 or df_within <= 0:
        raise ValueError("df_between and df_within must be > 0.")
    denom = df_between * f_stat + df_within
    return float(df_between * f_stat / denom) if denom > 0 else 0.0


def omega_squared(f_stat: float, df_between: int, df_within: int, n: int) -> float:
    """
    Omega-squared (omega^2) from ANOVA — less biased than eta-squared.

    omega^2 = (SS_between - df_between * MS_within) / (SS_total + MS_within)
            = (df_between * (F - 1)) / (df_between * F + df_within + 1)

    :param f_stat: F-statistic from one-way ANOVA.
    :param df_between: Between-groups df (k - 1).
    :param df_within: Within-groups df (N - k).
    :param n: Total sample size N.
    :return: Omega-squared; clipped to [0, 1] as negative values are set to 0.
    :raises ValueError: If f_stat < 0 or df values are non-positive.

    References
    ----------
    Hays, W. L. (1963). Statistics for Psychologists. Holt, Rinehart and Winston.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    if f_stat < 0:
        raise ValueError(f"f_stat must be >= 0, got {f_stat}.")
    if df_between <= 0 or df_within <= 0:
        raise ValueError("df_between and df_within must be > 0.")
    num = df_between * (f_stat - 1)
    denom = df_between * f_stat + df_within + 1
    return float(max(0.0, num / denom)) if denom > 0 else 0.0


def cramers_v(contingency_table: Union[list, np.ndarray]) -> float:
    """
    Cramér's V for categorical association (r x c contingency table).

    V = sqrt(chi^2 / (n * min(r-1, c-1)))

    Ranges from 0 (no association) to 1 (perfect association).
    Conventional benchmarks for df_min = 1: 0.10 small, 0.30 medium, 0.50 large.

    :param contingency_table: r x c contingency table (non-negative integers).
    :return: Cramér's V in [0, 1].
    :raises ValueError: If table has fewer than 2 rows or columns, or negative entries.

    References
    ----------
    Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    """
    tbl = np.asarray(contingency_table, dtype=float)
    if tbl.ndim != 2 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
        raise ValueError("contingency_table must be at least a 2x2 array.")
    if np.any(tbl < 0):
        raise ValueError("Table entries must be non-negative.")
    chi2_stat, _, _, _ = stats.chi2_contingency(tbl)
    n = tbl.sum()
    min_dim = min(tbl.shape[0] - 1, tbl.shape[1] - 1)
    denom = n * min_dim
    return float(math.sqrt(chi2_stat / denom)) if denom > 0 else 0.0


def phi_coefficient(table_2x2: Union[list, np.ndarray]) -> float:
    """
    Phi coefficient for association in a 2x2 contingency table.

    phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))

    Equivalent to the Pearson correlation between two binary variables.
    Ranges from -1 to +1.

    :param table_2x2: 2x2 array [[a, b], [c, d]].
    :return: Phi coefficient.
    :raises ValueError: If table is not 2x2 or marginal totals are zero.

    References
    ----------
    Pearson, K. (1900). Mathematical contributions to the theory of evolution.
        Philosophical Transactions of the Royal Society A, 195, 1-47.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float((a * d - b * c) / denom) if denom > 0 else 0.0


def point_biserial_r(
    binary_var: Union[list, np.ndarray],
    continuous_var: Union[list, np.ndarray],
) -> dict:
    """
    Point-biserial correlation between a binary and a continuous variable.

    r_pb = (M_1 - M_0) / s_total * sqrt(n_1 * n_0 / n^2)

    Equivalent to the Pearson correlation when one variable is dichotomous.

    :param binary_var: Binary variable (1-D array-like with values 0 and 1).
    :param continuous_var: Continuous variable (same length as binary_var).
    :return: dict with keys ``r``, ``p_value``.
    :raises ValueError: If inputs have different lengths or binary_var is not binary.

    References
    ----------
    Glass, G. V., & Hopkins, K. D. (1996). Statistical Methods in Education and
        Psychology (3rd ed.). Allyn & Bacon.
    """
    b = np.asarray(binary_var, dtype=float)
    c = np.asarray(continuous_var, dtype=float)
    if len(b) != len(c):
        raise ValueError("binary_var and continuous_var must have the same length.")
    unique_vals = set(np.unique(b))
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError(f"binary_var must contain only 0 and 1, found: {unique_vals}.")
    r, p_val = stats.pointbiserialr(b, c)
    return {"r": float(r), "p_value": float(p_val)}


def kendall_tau(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
) -> dict:
    """
    Kendall's tau-b rank correlation coefficient.

    A non-parametric measure of ordinal association that is less sensitive
    to outliers than Spearman's rho.

    :param x: First variable (1-D array-like).
    :param y: Second variable (same length as x).
    :return: dict with keys ``tau``, ``p_value``.
    :raises ValueError: If x and y have different lengths or fewer than 3 observations.

    References
    ----------
    Kendall, M. G. (1938). A new measure of rank correlation. Biometrika, 30(1-2), 81-93.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if len(ax) != len(ay):
        raise ValueError("x and y must have the same length.")
    if len(ax) < 3:
        raise ValueError("At least 3 observations are required.")
    tau, p_val = stats.kendalltau(ax, ay)
    return {"tau": float(tau), "p_value": float(p_val)}


def spearman_rho(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
) -> dict:
    """
    Spearman rank correlation coefficient (rho).

    Pearson correlation applied to ranks; captures monotone (not just linear)
    associations and is robust to outliers.

    :param x: First variable (1-D array-like).
    :param y: Second variable (same length as x).
    :return: dict with keys ``rho``, ``p_value``.
    :raises ValueError: If x and y have different lengths or fewer than 3 observations.

    References
    ----------
    Spearman, C. (1904). The proof and measurement of association between two things.
        American Journal of Psychology, 15(1), 72-101.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if len(ax) != len(ay):
        raise ValueError("x and y must have the same length.")
    if len(ax) < 3:
        raise ValueError("At least 3 observations are required.")
    rho, p_val = stats.spearmanr(ax, ay)
    return {"rho": float(rho), "p_value": float(p_val)}


# ===========================================================================
# SECTION 5 — POWER ANALYSIS
# ===========================================================================


def power_t_test(
    n: float | None = None,
    delta: float | None = None,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
    type: str = "two-sample",
) -> float:
    """
    Solve for any one missing parameter in a t-test power calculation.

    Exactly one of ``n``, ``delta``, or ``power`` must be None; the function
    solves for that parameter and returns it.

    Mirrors R's ``power.t.test()``.

    :param n: Sample size per group (two-sample) or total (one-sample).
    :param delta: Standardised effect size (|mean difference| / sd).
    :param sd: Standard deviation. Default 1.0.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power (1 - beta).
    :param alternative: ``"two-sided"`` or ``"one-sided"``. Default ``"two-sided"``.
    :param type: ``"two-sample"``, ``"one-sample"``, or ``"paired"``. Default ``"two-sample"``.
    :return: The value of the missing parameter.
    :raises ValueError: If exactly one parameter is not None, or invalid values provided.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    R Core Team (2024). power.t.test {stats}. R documentation.
    """
    none_count = sum(v is None for v in [n, delta, power])
    if none_count != 1:
        raise ValueError("Exactly one of n, delta, or power must be None.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if power is not None and not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")

    if alternative == "two-sided":
        ratio = 1 if type in ("one-sample", "paired") else 1
        two_tailed = True
    elif alternative in ("one-sided", "greater", "less"):
        two_tailed = False
    else:
        raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}.")

    # Effective effect size (Cohen's d-like) = delta / sd
    effect = (delta / sd) if delta is not None else None

    if type == "two-sample":
        analysis = TTestIndPower()
        ratio_arg = 1.0  # equal group sizes
    else:
        analysis = TTestPower()
        ratio_arg = None

    if n is None:
        if type == "two-sample":
            result = analysis.solve_power(
                effect_size=float(effect),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                effect_size=float(effect),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(result)
    elif delta is None:
        if type == "two-sample":
            result = analysis.solve_power(
                nobs1=float(n),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                nobs=float(n),
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(result) * float(sd)  # convert back to delta scale
    else:  # power is None
        if type == "two-sample":
            result = analysis.solve_power(
                effect_size=float(effect),
                nobs1=float(n),
                alpha=float(alpha),
                alternative="two-sided" if two_tailed else "larger",
                ratio=ratio_arg,
            )
        else:
            result = analysis.solve_power(
                effect_size=float(effect),
                nobs=float(n),
                alpha=float(alpha),
                alternative="two-sided" if two_tailed else "larger",
            )
        return float(np.clip(result, 0.0, 1.0))


def power_prop_test(
    n: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
) -> float:
    """
    Power for two-proportion z-test.

    Solves for one missing parameter among ``n``, ``p1``, ``p2``, or ``power``.
    Mirrors R's ``power.prop.test()``.

    :param n: Sample size per group.
    :param p1: Proportion in group 1.
    :param p2: Proportion in group 2.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power.
    :param alternative: ``"two-sided"`` or ``"one-sided"``. Default ``"two-sided"``.
    :return: The value of the missing parameter (n, or power).
    :raises ValueError: If p1 and p2 are both provided but either is out of [0, 1].

    Notes
    -----
    The NormalIndPower class operates on an arcsine-transformed effect size
    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)) (Cohen's h). This is the
    conventional approach for proportion tests.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
        Section 7. Effect size h.
    R Core Team (2024). power.prop.test {stats}. R documentation.
    """
    if p1 is not None and not 0 < p1 < 1:
        raise ValueError(f"p1 must be in (0, 1), got {p1}.")
    if p2 is not None and not 0 < p2 < 1:
        raise ValueError(f"p2 must be in (0, 1), got {p2}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    two_tailed = alternative == "two-sided"

    if p1 is not None and p2 is not None:
        # Cohen's h: effect size for proportions
        h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
        analysis = NormalIndPower()
        if n is None and power is not None:
            result = analysis.solve_power(
                effect_size=h,
                alpha=float(alpha),
                power=float(power),
                alternative="two-sided" if two_tailed else "larger",
            )
            return float(result)
        elif power is None and n is not None:
            result = analysis.solve_power(
                effect_size=h,
                alpha=float(alpha),
                nobs1=float(n),
                alternative="two-sided" if two_tailed else "larger",
            )
            return float(np.clip(result, 0.0, 1.0))
        else:
            raise ValueError("Provide exactly one of (n, power) when p1 and p2 are given.")
    else:
        raise ValueError("p1 and p2 must both be provided.")


def power_anova(
    n: float | None = None,
    k: int | None = None,
    f: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> float:
    """
    Power for one-way ANOVA. Solves for one missing parameter.

    Mirrors R's ``power.anova.test()``.

    :param n: Number of observations per group.
    :param k: Number of groups.
    :param f: Cohen's f effect size (f = sigma_between / sigma_within).
        Conventional benchmarks: 0.10 small, 0.25 medium, 0.40 large.
    :param alpha: Type I error rate. Default 0.05.
    :param power: Desired power.
    :return: The value of the missing parameter.
    :raises ValueError: If more than one parameter is None or k < 2.

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
        Section 8.
    R Core Team (2024). power.anova.test {stats}. R documentation.
    """
    none_count = sum(v is None for v in [n, k, f, power])
    if none_count != 1:
        raise ValueError("Exactly one of n, k, f, or power must be None.")
    if k is not None and k < 2:
        raise ValueError(f"k must be >= 2, got {k}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    analysis = FTestAnovaPower()

    if power is None:
        # nobs = total sample = n * k
        nobs = float(n * k)
        result = analysis.solve_power(
            effect_size=float(f),
            nobs=nobs,
            alpha=float(alpha),
            k_groups=int(k),
        )
        return float(np.clip(result, 0.0, 1.0))
    elif n is None:
        # Solve for n per group; FTestAnovaPower uses total nobs
        # We iterate: for a given k, find nobs then divide by k
        nobs = analysis.solve_power(
            effect_size=float(f),
            alpha=float(alpha),
            power=float(power),
            k_groups=int(k),
        )
        return float(nobs) / float(k)
    elif f is None:
        nobs = float(n * k)
        result = analysis.solve_power(
            nobs=nobs,
            alpha=float(alpha),
            power=float(power),
            k_groups=int(k),
        )
        return float(result)
    else:  # k is None — iterate over k (integer) to find smallest k meeting power
        # FTestAnovaPower requires integer k; we solve by scanning
        for k_try in range(2, 200):
            nobs = float(n * k_try)
            try:
                pwr = analysis.solve_power(
                    effect_size=float(f),
                    nobs=nobs,
                    alpha=float(alpha),
                    k_groups=k_try,
                )
                if float(pwr) >= float(power):
                    return float(k_try)
            except Exception:
                continue
        raise ValueError("Could not find k in [2, 200) achieving the desired power.")


def sample_size_logistic(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    power: float = 0.80,
    *,
    two_sided: bool = True,
) -> int:
    """
    Minimum sample size for logistic regression to detect a change in
    event probability from ``p0`` (control) to ``p1`` (treatment).

    Uses the Hsieh, Bloch & Larsen (1998) formula:

    .. math::

        n = \\frac{(z_{\\alpha/2} + z_{\\beta})^2}{p_0(1-p_0) + p_1(1-p_1)} \\cdot \\frac{(p_0 + p_1)}{2} \\cdot 2

    More precisely, this uses the arcsine transformation approach via
    Cohen's h for the two-proportion z-test, inflated by a factor to
    account for logistic regression efficiency relative to the z-test.

    :param p0: Event probability in the control/reference group (in (0, 1)).
    :param p1: Event probability in the intervention group (in (0, 1)).
    :param alpha: Type I error. Default 0.05.
    :param power: Desired power (1 - beta). Default 0.80.
    :param two_sided: If True use two-sided alpha. Default True.
    :return: Required total sample size (integer; equal allocation assumed).
    :raises ValueError: If p0 or p1 not in (0, 1).

    References
    ----------
    Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of
        sample size calculation for linear and logistic regression. Statistics
        in Medicine, 17(14), 1623-1634.
    """
    if not 0 < p0 < 1:
        raise ValueError(f"p0 must be in (0, 1), got {p0}.")
    if not 0 < p1 < 1:
        raise ValueError(f"p1 must be in (0, 1), got {p1}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")

    alpha_eff = alpha / 2 if two_sided else alpha
    z_alpha = float(stats.norm.ppf(1 - alpha_eff))
    z_beta = float(stats.norm.ppf(power))

    # Cohen's h effect size for the two proportions
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0)))
    # Per-group n from NormalIndPower (same as z-test approach)
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=h,
        alpha=float(alpha),
        power=float(power),
        alternative="two-sided" if two_sided else "larger",
    )
    # Inflate by logistic regression efficiency factor (~pi^2/3 variance inflation)
    # For equal groups: total = 2 * n_per_group (no additional inflation needed for OR approach)
    total_n = math.ceil(2 * n_per_group)
    return int(total_n)
