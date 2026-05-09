"""Tests for moirais.fn.vgcrf -- Correlogram function"""

import numpy as np
import pytest

from moirais.fn.vgcrf import correlogram


class TestCorrelogram:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = correlogram(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = correlogram(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
