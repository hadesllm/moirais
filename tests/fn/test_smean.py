"""Test sample_mean (smean)."""
import numpy as np
from moirais.fn.smean import sample_mean, smean
from moirais.fn._containers import DescriptiveResult


class TestSampleMean:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_mean(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "sample_mean"
        assert result.value == 3.0

    def test_negative(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert sample_mean(x).value == 0.0

    def test_alias(self):
        assert smean is sample_mean
