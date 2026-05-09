"""Tests for moirais.fn.kgorm -- Ordinary kriging matrix system"""

import numpy as np
import pytest

from moirais.fn.kgorm import ok_matrix


class TestOkMatrix:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ok_matrix(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ok_matrix(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
