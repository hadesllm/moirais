"""Tests for morie.fn.zelrx -- Leroux CAR model"""

import numpy as np
import pytest

from morie.fn.zelrx import leroux_model


class TestLerouxModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = leroux_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = leroux_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
