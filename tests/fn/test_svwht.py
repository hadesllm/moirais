"""Tests for morie.fn.svwht -- Wittman divergence model (policy-motivated)"""

import numpy as np
import pytest

from morie.fn.svwht import wittman_model


class TestWittmanModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wittman_model(data)
        assert result.value is not None

    def test_output_type(self):
        result = wittman_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
