"""Test confusion_plot (cmxpl)."""
import numpy as np
from moirais.fn.cmxpl import confusion_plot, cmxpl
from moirais.fn._containers import DescriptiveResult


class TestCmxpl:
    def test_basic(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        result = confusion_plot(y_true, y_pred)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "confusion_plot"
        cm = result.extra["confusion_matrix"]
        assert cm.shape == (2, 2)
        assert cm.sum() == 6

    def test_perfect(self):
        y = np.array([0, 1, 2, 0, 1])
        result = confusion_plot(y, y)
        assert result.extra["accuracy"] == 1.0

    def test_alias(self):
        assert cmxpl is confusion_plot
