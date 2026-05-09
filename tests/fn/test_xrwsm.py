"""Tests for moirais.fn.xrwsm -- Weights symmetrization"""

import numpy as np
import pytest

from moirais.fn.xrwsm import w_symmetrize


class TestWSymmetrize:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = w_symmetrize(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = w_symmetrize(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
