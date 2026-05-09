"""Test mobility (smobl)."""
import numpy as np
from moirais.fn.smobl import mobility, smobl
from moirais.fn._containers import DescriptiveResult


class TestMobility:
    def test_positive(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = mobility(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0

    def test_constant(self):
        x = np.ones(10)
        assert mobility(x).value == 0.0

    def test_alias(self):
        assert smobl is mobility
