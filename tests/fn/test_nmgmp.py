"""Tests for morie.fn.nmgmp -- Geometric Mean Probability"""

import numpy as np
import pytest

from morie.fn.nmgmp import gmp_stat


class TestGmpStat:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gmp_stat(data)
        assert result.value is not None

    def test_output_type(self):
        result = gmp_stat(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
