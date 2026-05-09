"""Test baseline_corrected_correlation."""
import numpy as np
from moirais.fn.bscor import baseline_corrected_correlation, bscor
from moirais.fn._containers import DescriptiveResult


class TestBaselineCorrectedCorrelation:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(100)
        y = np.random.default_rng(43).standard_normal(100)
        result = baseline_corrected_correlation(x, y)
        assert isinstance(result, DescriptiveResult)

    def test_perfect_correlation(self):
        x = np.arange(100, dtype=float)
        result = baseline_corrected_correlation(x, x)
        assert abs(result.value - 1.0) < 1e-9

    def test_range(self):
        x = np.random.default_rng(42).standard_normal(100)
        y = np.random.default_rng(43).standard_normal(100)
        result = baseline_corrected_correlation(x, y)
        assert -1.0 <= result.value <= 1.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(100)
        y = np.random.default_rng(43).standard_normal(100)
        result = baseline_corrected_correlation(x, y)
        assert result.name == "baseline_corrected_corr"

    def test_alias(self):
        assert bscor is baseline_corrected_correlation
