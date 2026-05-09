"""Tests for moirais.fn.hswas."""

import numpy as np
import pytest
from moirais.fn.hswas import hswas


class TestHswas:
    def test_basic(self):
        result = hswas(-0.3, 0.3)
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_spatial_result(self):
        result = hswas(-0.3, 0.3)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = hswas(-0.3, 0.3)
        assert np.isfinite(result.value)

    def test_name_string(self):
        result = hswas(-0.3, 0.3)
        assert isinstance(result.name, str)
        assert len(result.name) > 0
