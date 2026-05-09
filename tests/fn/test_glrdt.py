"""Test glr_detector (glrdt)."""
import numpy as np
from moirais.fn.glrdt import glr_detector, glrdt
from moirais.fn._containers import DescriptiveResult


class TestGlrDetector:
    def test_basic(self):
        x = np.concatenate([np.zeros(100), np.ones(100)])
        result = glr_detector(x, window=20)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "glr_detector"

    def test_detects_change(self):
        x = np.concatenate([np.zeros(100), np.ones(100) * 5])
        result = glr_detector(x, window=20)
        cp = result.extra["change_point"]
        assert 80 <= cp <= 120

    def test_positive_statistic(self):
        x = np.concatenate([np.zeros(100), np.ones(100)])
        result = glr_detector(x, window=20)
        assert result.value > 0

    def test_alias(self):
        assert glrdt is glr_detector
