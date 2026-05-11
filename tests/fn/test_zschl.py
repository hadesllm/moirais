"""Tests for morie.fn.zschl -- Cholesky spatial simulation"""

import numpy as np
import pytest

from morie.fn.zschl import chol_sim


class TestCholSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = chol_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = chol_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
