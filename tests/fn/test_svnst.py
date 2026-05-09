"""Tests for moirais.fn.svnst."""

import numpy as np
import pytest
from moirais.fn.svnst import svnst


class TestSvnst:
    def test_basic(self):
        result = svnst(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert result is not None
        assert result.value is not None
        assert isinstance(result.value, float)

    def test_returns_descriptive_result(self):
        result = svnst(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert hasattr(result, "value")
        assert hasattr(result, "method")
        assert hasattr(result, "extra")

    def test_finite_output(self):
        result = svnst(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert np.isfinite(result.value)

    def test_method_string(self):
        result = svnst(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        assert isinstance(result.method, str)
        assert len(result.method) > 0
