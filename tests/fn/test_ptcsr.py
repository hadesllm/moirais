"""Tests for moirais.fn.ptcsr -- Complete Spatial Randomness test"""

import numpy as np
import pytest

from moirais.fn.ptcsr import csr_test


class TestCsrTest:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = csr_test(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = csr_test(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
