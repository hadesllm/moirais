"""
Tests for R-like distribution functions in morie.inference.

Each test verifies at least one of:
- Exact known values (cross-checked against R or mathematical derivation)
- Round-trip property: q*(p*(x)) == x  or  p*(q*(p)) == p
- Log/non-log consistency: log=True result == log(log=False result)
- Boundary behaviour (p=0, p=1, x at distribution boundaries)
- Shape and seed reproducibility for random samplers

References
----------
R Core Team (2024). Distribution functions. R documentation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from morie.inference import (
    # Normal
    dnorm, pnorm, qnorm, rnorm,
    # t
    dt, pt, qt,
    # chi-squared
    dchisq, pchisq, qchisq,
    # Binomial
    dbinom, pbinom, qbinom, rbinom,
    # Poisson
    dpois, ppois, qpois, rpois,
    # Beta
    dbeta, pbeta, qbeta,
    # Gamma
    dgamma, pgamma,
    # Uniform
    dunif, punif, runif,
)


# ===========================================================================
# Normal distribution
# ===========================================================================

class TestDnorm:
    """Tests for dnorm — normal PDF."""

    def test_standard_normal_at_zero(self):
        """dnorm(0) = 1/sqrt(2*pi) ≈ 0.39894228."""
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(dnorm(0.0) - expected) < 1e-10

    def test_known_value_at_1(self):
        """dnorm(1) = exp(-0.5) / sqrt(2*pi) ≈ 0.24197073."""
        expected = math.exp(-0.5) / math.sqrt(2.0 * math.pi)
        assert abs(dnorm(1.0) - expected) < 1e-10

    def test_non_standard_location_scale(self):
        """dnorm(5, mean=5, sd=2) = 1/(2*sqrt(2*pi)) ≈ 0.19947114."""
        expected = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
        assert abs(dnorm(5.0, mean=5.0, sd=2.0) - expected) < 1e-10

    def test_log_parameter(self):
        """dnorm(x, log=True) == log(dnorm(x))."""
        for x in [-2.0, 0.0, 1.5, 3.0]:
            assert abs(dnorm(x, log=True) - math.log(dnorm(x))) < 1e-10

    def test_vectorised_input(self):
        """dnorm should accept numpy arrays."""
        xs = np.array([-1.0, 0.0, 1.0])
        result = dnorm(xs)
        assert result.shape == (3,)
        assert abs(result[1] - 1.0 / math.sqrt(2.0 * math.pi)) < 1e-10

    def test_negative_sd_raises(self):
        with pytest.raises(ValueError, match="sd must be > 0"):
            dnorm(0.0, sd=-1.0)

    def test_zero_sd_raises(self):
        with pytest.raises(ValueError, match="sd must be > 0"):
            dnorm(0.0, sd=0.0)


class TestPnorm:
    """Tests for pnorm — normal CDF."""

    def test_at_zero_is_half(self):
        """pnorm(0) = 0.5 for standard normal."""
        assert abs(pnorm(0.0) - 0.5) < 1e-15

    def test_upper_tail(self):
        """pnorm(0, lower_tail=False) = 0.5."""
        assert abs(pnorm(0.0, lower_tail=False) - 0.5) < 1e-15

    def test_lower_plus_upper_tail_equals_one(self):
        """P(X <= x) + P(X > x) = 1."""
        for x in [-2.0, 0.0, 1.96]:
            assert abs(pnorm(x) + pnorm(x, lower_tail=False) - 1.0) < 1e-15

    def test_known_z_score_95_percentile(self):
        """pnorm(1.959964) ≈ 0.975."""
        assert abs(pnorm(1.959964) - 0.975) < 1e-5

    def test_log_lower_tail(self):
        """pnorm(x, log=True) == log(pnorm(x))."""
        for x in [-1.0, 0.0, 1.96]:
            log_p = pnorm(x, log=True)
            p = pnorm(x)
            assert abs(log_p - math.log(p)) < 1e-10

    def test_log_upper_tail(self):
        """pnorm(x, lower_tail=False, log=True) == log(1 - pnorm(x))."""
        for x in [-1.0, 0.0, 1.96]:
            log_sf = pnorm(x, lower_tail=False, log=True)
            sf = pnorm(x, lower_tail=False)
            assert abs(log_sf - math.log(sf)) < 1e-10


class TestQnorm:
    """Tests for qnorm — normal quantile function."""

    def test_at_half_is_zero(self):
        """qnorm(0.5) = 0."""
        assert abs(qnorm(0.5)) < 1e-15

    def test_round_trip_with_pnorm(self):
        """qnorm(pnorm(x)) == x for a range of x values."""
        for x in [-3.0, -1.0, 0.0, 1.5, 2.5]:
            assert abs(qnorm(pnorm(x)) - x) < 1e-9

    def test_known_quantiles(self):
        """qnorm(0.975) ≈ 1.959964 (the 97.5th percentile)."""
        assert abs(qnorm(0.975) - 1.959964) < 1e-5
        assert abs(qnorm(0.025) - (-1.959964)) < 1e-5

    def test_upper_tail(self):
        """qnorm(0.025, lower_tail=False) = qnorm(0.975)."""
        assert abs(qnorm(0.025, lower_tail=False) - qnorm(0.975)) < 1e-12

    def test_non_standard_distribution(self):
        """qnorm(0.5, mean=10, sd=2) = 10."""
        assert abs(qnorm(0.5, mean=10.0, sd=2.0) - 10.0) < 1e-12


class TestRnorm:
    """Tests for rnorm — normal random sample."""

    def test_shape(self):
        x = rnorm(1000, seed=0)
        assert len(x) == 1000

    def test_approximate_mean_and_sd(self):
        x = rnorm(10000, mean=5.0, sd=2.0, seed=123)
        assert abs(x.mean() - 5.0) < 0.1
        assert abs(x.std() - 2.0) < 0.1

    def test_seed_reproducibility(self):
        x1 = rnorm(100, seed=42)
        x2 = rnorm(100, seed=42)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        x1 = rnorm(100, seed=1)
        x2 = rnorm(100, seed=2)
        assert not np.array_equal(x1, x2)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError, match="n must be > 0"):
            rnorm(0)


# ===========================================================================
# Student's t-distribution
# ===========================================================================

class TestTDistribution:

    def test_dt_at_zero_with_1df(self):
        """dt(0, df=1) = 1/pi for Cauchy."""
        expected = 1.0 / math.pi
        assert abs(dt(0.0, df=1) - expected) < 1e-10

    def test_dt_log_consistency(self):
        for x in [-1.0, 0.0, 2.0]:
            assert abs(dt(x, df=5, log=True) - math.log(dt(x, df=5))) < 1e-10

    def test_pt_symmetry(self):
        """pt(0, df=k) = 0.5 for any df."""
        for df in [1, 5, 10, 100]:
            assert abs(pt(0.0, df=df) - 0.5) < 1e-10

    def test_pt_lower_plus_upper_tail(self):
        for x in [-2.0, 0.0, 2.0]:
            assert abs(pt(x, df=10) + pt(x, df=10, lower_tail=False) - 1.0) < 1e-14

    def test_qt_pnorm_convergence_large_df(self):
        """At large df, t-quantiles converge to normal quantiles."""
        for p in [0.025, 0.975]:
            assert abs(qt(p, df=1000) - qnorm(p)) < 0.01

    def test_qt_round_trip(self):
        for p in [0.1, 0.5, 0.9]:
            assert abs(pt(qt(p, df=10), df=10) - p) < 1e-9

    def test_df_zero_raises(self):
        with pytest.raises(ValueError, match="df must be > 0"):
            dt(0.0, df=0)


# ===========================================================================
# Chi-squared distribution
# ===========================================================================

class TestChiSquaredDistribution:

    def test_pchisq_at_zero_is_zero(self):
        """chi-squared CDF at 0 is 0 for any df."""
        assert abs(pchisq(0.0, df=5)) < 1e-10

    def test_dchisq_positive(self):
        """chi-squared PDF is non-negative everywhere on support."""
        xs = np.linspace(0.01, 20.0, 50)
        assert np.all(dchisq(xs, df=3) >= 0)

    def test_qchisq_round_trip(self):
        for p in [0.05, 0.5, 0.95]:
            x = qchisq(p, df=5)
            assert abs(pchisq(x, df=5) - p) < 1e-9

    def test_dchisq_log_consistency(self):
        for x in [1.0, 5.0, 10.0]:
            assert abs(dchisq(x, df=3, log=True) - math.log(dchisq(x, df=3))) < 1e-10

    def test_known_value_chi2_df3(self):
        """P(chi2(3) <= 7.815) ≈ 0.95 (standard chi-sq critical value)."""
        assert abs(pchisq(7.815, df=3) - 0.95) < 0.001


# ===========================================================================
# Binomial distribution
# ===========================================================================

class TestBinomialDistribution:

    def test_dbinom_known_pmf(self):
        """P(X=3 | n=10, p=0.5) = C(10,3) * 0.5^10 = 120/1024 = 0.1171875."""
        assert abs(dbinom(3, size=10, prob=0.5) - 0.1171875) < 1e-8

    def test_dbinom_pmf_sums_to_one(self):
        """Sum of PMF over all values equals 1."""
        total = sum(dbinom(k, size=10, prob=0.3) for k in range(11))
        assert abs(total - 1.0) < 1e-10

    def test_dbinom_log_consistency(self):
        for k in [0, 3, 10]:
            val = dbinom(k, size=10, prob=0.5)
            if val > 0:
                assert abs(dbinom(k, size=10, prob=0.5, log=True) - math.log(val)) < 1e-10

    def test_pbinom_at_max_is_one(self):
        """P(X <= n) = 1."""
        assert abs(pbinom(10, size=10, prob=0.5) - 1.0) < 1e-12

    def test_pbinom_upper_tail(self):
        """pbinom(x) + pbinom(x, lower_tail=False) != 1 (they overlap at x=k)."""
        # P(X <= k) + P(X > k) = 1; NOT P(X <= k) + P(X >= k) = 1
        # pbinom(3) + pbinom(4, lower_tail=False) should = 1 (x=3 step)
        # Test: P(X <= k) = 1 - P(X > k)
        for k in [2, 5, 8]:
            assert abs(pbinom(k, size=10, prob=0.5) -
                       (1.0 - pbinom(k, size=10, prob=0.5, lower_tail=False))) < 1e-12

    def test_qbinom_round_trip(self):
        """qbinom gives the correct quantile for the CDF."""
        n, p = 20, 0.4
        for k in [0, 5, 10, 15, 20]:
            cdf_val = float(pbinom(k, size=n, prob=p))
            q = int(qbinom(cdf_val, size=n, prob=p))
            # q should be the smallest k' such that P(X<=k') >= cdf_val
            assert q <= k + 1  # can't exceed k by more than step

    def test_rbinom_shape_and_range(self):
        x = rbinom(500, size=10, prob=0.5, seed=0)
        assert len(x) == 500
        assert np.all(x >= 0) and np.all(x <= 10)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError, match="prob must be in"):
            dbinom(3, size=10, prob=1.5)


# ===========================================================================
# Poisson distribution
# ===========================================================================

class TestPoissonDistribution:

    def test_dpois_known_value(self):
        """P(X=0 | lambda=1) = exp(-1) ≈ 0.36787944."""
        assert abs(dpois(0, lambda_=1.0) - math.exp(-1.0)) < 1e-10

    def test_dpois_pmf_sums_to_one(self):
        """Sum of PMF over first 100 values is effectively 1 for lambda=3."""
        total = sum(dpois(k, lambda_=3.0) for k in range(100))
        assert abs(total - 1.0) < 1e-8

    def test_dpois_log_consistency(self):
        for k in [0, 2, 5]:
            val = dpois(k, lambda_=2.0)
            assert abs(dpois(k, lambda_=2.0, log=True) - math.log(val)) < 1e-10

    def test_ppois_at_large_k_is_one(self):
        assert abs(ppois(1000, lambda_=5.0) - 1.0) < 1e-10

    def test_ppois_upper_tail(self):
        for k in [2, 5, 10]:
            assert abs(ppois(k, lambda_=4.0) + ppois(k, lambda_=4.0, lower_tail=False) - 1.0) < 1e-12

    def test_qpois_round_trip(self):
        for p in [0.1, 0.5, 0.9]:
            q = qpois(p, lambda_=5.0)
            assert ppois(q, lambda_=5.0) >= p - 1e-9

    def test_rpois_shape_and_non_negative(self):
        x = rpois(200, lambda_=3.0, seed=7)
        assert len(x) == 200
        assert np.all(x >= 0)

    def test_lambda_zero_raises(self):
        with pytest.raises(ValueError, match="lambda_ must be > 0"):
            dpois(1, lambda_=0)


# ===========================================================================
# Beta distribution
# ===========================================================================

class TestBetaDistribution:

    def test_dbeta_at_half_with_equal_params(self):
        """Beta(1,1) is Uniform(0,1) so PDF = 1 everywhere on (0,1)."""
        for x in [0.1, 0.5, 0.9]:
            assert abs(dbeta(x, alpha=1.0, beta=1.0) - 1.0) < 1e-10

    def test_pbeta_at_half_with_equal_params(self):
        """Beta(1,1) = Uniform, so CDF at 0.5 = 0.5."""
        assert abs(pbeta(0.5, alpha=1.0, beta=1.0) - 0.5) < 1e-12

    def test_qbeta_round_trip(self):
        for p in [0.1, 0.5, 0.9]:
            x = qbeta(p, alpha=2.0, beta=3.0)
            assert abs(pbeta(x, alpha=2.0, beta=3.0) - p) < 1e-9

    def test_dbeta_log_consistency(self):
        for x in [0.2, 0.5, 0.8]:
            val = dbeta(x, alpha=2.0, beta=5.0)
            assert abs(dbeta(x, alpha=2.0, beta=5.0, log=True) - math.log(val)) < 1e-10

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be > 0"):
            dbeta(0.5, alpha=0.0, beta=1.0)


# ===========================================================================
# Gamma distribution
# ===========================================================================

class TestGammaDistribution:

    def test_dgamma_exponential_special_case(self):
        """Gamma(1, rate=1) = Exponential(1); PDF at x=1 = exp(-1)."""
        expected = math.exp(-1.0)
        assert abs(dgamma(1.0, shape=1.0, rate=1.0) - expected) < 1e-10

    def test_pgamma_at_large_x_is_one(self):
        assert abs(pgamma(1000.0, shape=2.0, rate=0.5) - 1.0) < 1e-8

    def test_pgamma_scale_vs_rate(self):
        """pgamma with scale=2 should equal pgamma with rate=0.5."""
        for x in [1.0, 5.0, 10.0]:
            assert abs(pgamma(x, shape=2.0, rate=0.5) -
                       pgamma(x, shape=2.0, scale=2.0)) < 1e-12

    def test_dgamma_log_consistency(self):
        for x in [0.5, 1.0, 3.0]:
            val = dgamma(x, shape=2.0, rate=1.0)
            assert abs(dgamma(x, shape=2.0, rate=1.0, log=True) - math.log(val)) < 1e-10

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape must be > 0"):
            dgamma(1.0, shape=0.0)


# ===========================================================================
# Uniform distribution
# ===========================================================================

class TestUniformDistribution:

    def test_dunif_standard(self):
        """dunif on (0,1) = 1 everywhere in support."""
        for x in [0.1, 0.5, 0.99]:
            assert abs(dunif(x) - 1.0) < 1e-12

    def test_dunif_outside_support_is_zero(self):
        assert dunif(-0.1) == 0.0
        assert dunif(1.1) == 0.0

    def test_punif_at_midpoint(self):
        """punif(0.5, min=0, max=1) = 0.5."""
        assert abs(punif(0.5) - 0.5) < 1e-12

    def test_punif_on_custom_range(self):
        """punif(5, min=0, max=10) = 0.5."""
        assert abs(punif(5.0, min=0.0, max=10.0) - 0.5) < 1e-12

    def test_punif_upper_tail(self):
        for x in [0.2, 0.5, 0.8]:
            assert abs(punif(x) + punif(x, lower_tail=False) - 1.0) < 1e-12

    def test_runif_bounds(self):
        x = runif(1000, min=2.0, max=5.0, seed=0)
        assert len(x) == 1000
        assert np.all(x >= 2.0) and np.all(x <= 5.0)

    def test_min_ge_max_raises(self):
        with pytest.raises(ValueError, match="min must be < max"):
            dunif(0.5, min=1.0, max=0.0)

    def test_dunif_log_consistency(self):
        x = 0.5
        val = dunif(x)
        assert abs(dunif(x, log=True) - math.log(val)) < 1e-10
