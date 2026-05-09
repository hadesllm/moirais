"""Tests for moirais.fn.zeism -- Indirect standardization"""

import numpy as np
import pytest

from moirais.fn.zeism import indirect_std


class TestIndirectStd:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = indirect_std(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = indirect_std(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
