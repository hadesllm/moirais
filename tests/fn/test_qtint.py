"""Test qt_interval (qtint)."""
import numpy as np
from moirais.fn.qtint import qt_interval, qtint
from moirais.fn._containers import DescriptiveResult


class TestQtInterval:
    def test_basic(self):
        qrs_on = np.array([100, 500, 900])
        t_off = np.array([200, 600, 1000])
        result = qt_interval(qrs_on, t_off, fs=250.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "qt_interval"

    def test_correct_duration(self):
        qrs_on = np.array([0, 100])
        t_off = np.array([50, 150])
        result = qt_interval(qrs_on, t_off, fs=100.0)
        assert np.allclose(result.value, 0.5)

    def test_empty(self):
        result = qt_interval(np.array([]), np.array([]), fs=1.0)
        assert result.value == 0.0

    def test_alias(self):
        assert qtint is qt_interval
