"""Test kaiming_init."""
import numpy as np
from morie.fn.kaimg import kaiming_init, kaimg
from morie.fn._containers import DescriptiveResult


class TestKaimingInit:
    def test_basic(self):
        result = kaiming_init(64, 128, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "kaiming_init"

    def test_shape(self):
        result = kaiming_init(64, 128, seed=42)
        assert result.extra["weights"].shape == (128, 64)

    def test_std_approx(self):
        result = kaiming_init(1024, 1024, seed=42)
        expected_std = np.sqrt(2.0 / 1024)
        actual_std = np.std(result.extra["weights"])
        assert abs(actual_std - expected_std) < 0.01

    def test_alias(self):
        assert kaimg is kaiming_init
