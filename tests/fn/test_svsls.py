"""Tests for morie.fn.svsls -- Issue salience weighted model"""

import numpy as np
import pytest

from morie.fn.svsls import salience_model


class TestSalienceModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = salience_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = salience_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
