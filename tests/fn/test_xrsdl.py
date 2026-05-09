"""Tests for moirais.fn.xrsdl -- Spatial Durbin Error model"""

import numpy as np
import pytest

from moirais.fn.xrsdl import sdem_ml


class TestSdemMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sdem_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = sdem_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
