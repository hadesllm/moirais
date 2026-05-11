"""Tests for morie.fn.xricar -- Intrinsic CAR (ICAR)"""

import numpy as np
import pytest

from morie.fn.xricar import icar_model


class TestIcarModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = icar_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = icar_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
