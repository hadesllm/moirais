"""Tests for moirais.fn.hdi -- Highest Density Interval."""

import numpy as np
import pytest
from moirais.fn.hdi import highest_density_interval


class TestHDI:
    def test_symmetric_distribution(self):
        """For symmetric data, HDI should be roughly symmetric."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        lo, hi = highest_density_interval(samples, mass=0.95)
        assert lo < 0 < hi
        # Roughly symmetric around 0
        assert abs(lo + hi) < 0.2

    def test_interval_contains_mode(self):
        """HDI should contain the mode of a unimodal distribution."""
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 1.0, 5000)
        lo, hi = highest_density_interval(samples, mass=0.90)
        assert lo < 5.0 < hi

    def test_mass_coverage(self):
        """Approximately mass% of samples should fall in the HDI."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 10000)
        lo, hi = highest_density_interval(samples, mass=0.95)
        coverage = np.mean((samples >= lo) & (samples <= hi))
        assert coverage >= 0.94  # allow small tolerance

    def test_invalid_mass_raises(self):
        with pytest.raises(ValueError, match="mass"):
            highest_density_interval([1, 2, 3, 4], mass=1.5)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError):
            highest_density_interval([1, 2], mass=0.95)
