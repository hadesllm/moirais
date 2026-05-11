"""Tests for morie.fn.rcsca."""

import numpy as np
import pytest
from morie.fn.rcsca import rcsca


class TestRcsca:
    def test_basic(self):
        result = rcsca(0.5, 0.0)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = rcsca(0.5, 0.0)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = rcsca(0.5, 0.0)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = rcsca(0.5, 0.0)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
