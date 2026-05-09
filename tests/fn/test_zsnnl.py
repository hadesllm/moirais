"""Tests for moirais.fn.zsnnl -- Laplace natural neighbor (Sibson)"""

import numpy as np
import pytest

from moirais.fn.zsnnl import nn_laplace


class TestNnLaplace:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = nn_laplace(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = nn_laplace(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
