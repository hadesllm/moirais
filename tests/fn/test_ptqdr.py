"""Tests for moirais.fn.ptqdr -- Quadrat count test"""

import numpy as np
import pytest

from moirais.fn.ptqdr import quadrat_test


class TestQuadratTest:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quadrat_test(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = quadrat_test(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
