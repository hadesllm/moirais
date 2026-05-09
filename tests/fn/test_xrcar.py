"""Tests for moirais.fn.xrcar -- CAR model ML"""

import numpy as np
import pytest

from moirais.fn.xrcar import car_ml


class TestCarMl:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = car_ml(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = car_ml(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
