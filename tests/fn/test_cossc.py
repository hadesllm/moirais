"""Test cost_sensitive (cossc)."""
import numpy as np
from moirais.fn.cossc import cost_sensitive, cossc
from moirais.fn._containers import DescriptiveResult


class TestCossc:
    def test_basic(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = cost_sensitive(y_true, y_pred)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cost_sensitive"
        assert result.extra["total_cost"] > 0

    def test_perfect(self):
        y = np.array([0, 1, 0, 1])
        result = cost_sensitive(y, y)
        assert result.extra["total_cost"] == 0

    def test_alias(self):
        assert cossc is cost_sensitive
