"""Tests for moirais.fn.mpctr."""

import numpy as np
import pytest
from moirais.fn.mpctr import mpctr


class TestMpctr:
    def test_basic(self):
        result = mpctr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = mpctr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = mpctr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = mpctr(np.array([-0.8, -0.2, 0.3, 0.9]))
        assert isinstance(result.name, str)
        assert len(result.name) > 0
