"""Tests for morie.fn.hsmlt."""

import numpy as np
import pytest
from morie.fn.hsmlt import hsmlt


class TestHsmlt:
    def test_basic(self):
        result = hsmlt(-0.3, 0.3)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = hsmlt(-0.3, 0.3)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = hsmlt(-0.3, 0.3)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = hsmlt(-0.3, 0.3)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
