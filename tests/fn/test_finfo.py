"""Tests for morie.fn.finfo — Fisher information matrix."""

import numpy as np
import pytest

from morie.fn.finfo import finfo


class TestFinfo:
    def test_normal_mean_fisher(self):
        def score(x, theta):
            return np.array([(x - theta[0]) / 1.0])

        def gen(n, theta, rng):
            return rng.normal(theta[0], 1.0, size=n)

        result = finfo(score, np.array([0.0]), data_generator=gen,
                       n_samples=10000, seed=42)
        assert result["fisher_info"][0, 0] == pytest.approx(1.0, abs=0.15)

    def test_crlb_inverse_of_fim(self):
        def score(x, theta):
            return np.array([(x - theta[0]) / 4.0])

        def gen(n, theta, rng):
            return rng.normal(theta[0], 2.0, size=n)

        result = finfo(score, np.array([0.0]), data_generator=gen,
                       n_samples=5000, seed=42)
        assert result["crlb"][0] > 0

    def test_no_generator_error(self):
        with pytest.raises(ValueError):
            finfo(lambda x, t: x, np.array([1.0]))

    def test_output_keys(self):
        def score(x, theta):
            return np.array([x - theta[0]])

        def gen(n, theta, rng):
            return rng.normal(theta[0], 1.0, size=n)

        result = finfo(score, np.array([0.0]), data_generator=gen, n_samples=100)
        assert "fisher_info" in result
        assert "crlb" in result
