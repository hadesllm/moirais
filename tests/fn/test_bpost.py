"""Tests for morie.fn.bpost -- Conjugate posterior updating."""

import numpy as np
import pytest
from morie.fn.bpost import conjugate_posterior


class TestConjugatePosterior:
    def test_normal_posterior_mean_shifts(self):
        """Posterior mean should shift toward data mean."""
        data = [5.0, 5.5, 4.5, 5.2, 4.8]
        result = conjugate_posterior(data, model="normal", prior_params={"mu": 0.0, "var": 1.0})
        assert result["posterior_mean"] > 0.0  # shifted from prior=0 toward data~5

    def test_normal_posterior_precision_increases(self):
        """Posterior variance should be less than prior variance."""
        data = [1.0, 1.1, 0.9]
        result = conjugate_posterior(data, model="normal", prior_params={"mu": 0.0, "var": 10.0})
        assert result["posterior_sd"] < np.sqrt(10.0)

    def test_beta_binomial_known_values(self):
        """Beta(1,1) + 7 successes in 10 trials -> Beta(8, 4)."""
        data = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        result = conjugate_posterior(data, model="beta_binomial", prior_params={"a": 1.0, "b": 1.0})
        assert result["posterior_params"]["a"] == pytest.approx(8.0)
        assert result["posterior_params"]["b"] == pytest.approx(4.0)

    def test_credible_interval_brackets_mean(self):
        """CI should contain the posterior mean."""
        data = [2.0, 3.0, 2.5]
        result = conjugate_posterior(data, model="normal")
        lo, hi = result["credible_interval"]
        assert lo <= result["posterior_mean"] <= hi

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            conjugate_posterior([1, 2], model="poisson")

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            conjugate_posterior([], model="normal")
