"""Test snr_improvement (ssnri)."""
import numpy as np
import pytest
from moirais.fn.ssnri import snr_improvement, ssnri
from moirais.fn._containers import DescriptiveResult


class TestSNRImprovement:
    def test_basic(self):
        result = snr_improvement(4)
        assert isinstance(result, DescriptiveResult)
        assert abs(result.value - 2.0) < 1e-10

    def test_one(self):
        assert abs(snr_improvement(1).value - 1.0) < 1e-10

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            snr_improvement(-1)

    def test_alias(self):
        assert ssnri is snr_improvement
