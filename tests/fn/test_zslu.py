"""Tests for morie.fn.zslu -- LU decomposition simulation"""

import numpy as np
import pytest

from morie.fn.zslu import lu_sim


class TestLuSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lu_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = lu_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
