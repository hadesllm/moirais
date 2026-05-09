"""Test qrs_duration (qrsdr)."""
import numpy as np
from moirais.fn.qrsdr import qrs_duration, qrsdr
from moirais.fn._containers import DescriptiveResult


class TestQrsDuration:
    def test_basic(self):
        qrs_on = np.array([100, 500, 900])
        qrs_off = np.array([130, 530, 930])
        result = qrs_duration(qrs_on, qrs_off, fs=250.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "qrs_duration"

    def test_correct_duration(self):
        qrs_on = np.array([0, 100])
        qrs_off = np.array([25, 125])
        result = qrs_duration(qrs_on, qrs_off, fs=250.0)
        assert np.allclose(result.value, 0.1)

    def test_empty(self):
        result = qrs_duration(np.array([]), np.array([]), fs=1.0)
        assert result.value == 0.0

    def test_alias(self):
        assert qrsdr is qrs_duration
