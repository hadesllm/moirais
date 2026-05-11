"""Tests for morie.fn.nmpre -- Proportional Reduction in Error"""

import numpy as np
import pytest

from morie.fn.nmpre import pre_stat


class TestPreStat:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pre_stat(data)
        assert result.value is not None

    def test_output_type(self):
        result = pre_stat(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
