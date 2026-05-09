"""Tests for moirais.fn.kgsmv -- Simple kriging variance"""

import numpy as np
import pytest

from moirais.fn.kgsmv import sk_variance


class TestSkVariance:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sk_variance(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sk_variance(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
