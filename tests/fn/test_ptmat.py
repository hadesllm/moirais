"""Tests for morie.fn.ptmat -- Matern cluster process"""

import numpy as np
import pytest

from morie.fn.ptmat import matern_process


class TestMaternProcess:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = matern_process(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = matern_process(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
