"""Tests for moirais.fn.svrrd -- Roemer party unanimity model"""

import numpy as np
import pytest

from moirais.fn.svrrd import roemer_model


class TestRoemerModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = roemer_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = roemer_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
