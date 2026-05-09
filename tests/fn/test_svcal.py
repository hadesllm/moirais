"""Tests for moirais.fn.svcal -- Calvert uncertainty model"""

import numpy as np
import pytest

from moirais.fn.svcal import calvert_model


class TestCalvertModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calvert_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = calvert_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
