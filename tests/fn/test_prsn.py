"""Test pearson_corr (prsn)."""
import numpy as np
from moirais.fn.prsn import pearson_corr, prsn
from moirais.fn._containers import DescriptiveResult


class TestPearsonCorr:
    def test_perfect(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = pearson_corr(x, x)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 1.0) < 1e-10

    def test_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 2.0, 1.0])
        assert abs(pearson_corr(x, y).value - (-1.0)) < 1e-10

    def test_alias(self):
        assert prsn is pearson_corr
