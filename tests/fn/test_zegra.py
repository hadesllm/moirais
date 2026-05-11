"""Tests for morie.fn.zegra -- Gravity-based accessibility"""

import numpy as np
import pytest

from morie.fn.zegra import gravity_access


class TestGravityAccess:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = gravity_access(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = gravity_access(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
