"""Tests for moirais.fn.cv -- coefficient of variation."""

import numpy as np
import pytest
from moirais.fn.cv import coefficient_of_variation


class TestCoefficientOfVariation:
    def test_known_values(self):
        """Mean=10, sd=2 -> CV = 2/10 = 0.2."""
        # Construct data with known mean and sd
        x = [8.0, 10.0, 12.0]  # mean=10, sd=2
        result = coefficient_of_variation(x)
        assert result.measure == "Coefficient of variation"
        assert result.estimate == pytest.approx(0.2, rel=0.01)

    def test_zero_mean(self):
        """Zero-mean data should give inf CV."""
        x = [-1.0, 0.0, 1.0]
        result = coefficient_of_variation(x)
        assert result.estimate == np.inf or result.estimate > 1e6

    def test_constant_data(self):
        """Constant data has sd=0, so CV=0."""
        x = [5.0, 5.0, 5.0]
        result = coefficient_of_variation(x)
        assert result.estimate == pytest.approx(0.0, abs=1e-10)
