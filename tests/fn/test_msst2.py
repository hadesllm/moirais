"""Tests for moirais.fn.msst2 -- Normalized stress"""

import numpy as np
import pytest

from moirais.fn.msst2 import stress_norm


class TestStressNorm:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = stress_norm(X)
        assert result.value is not None

    def test_output_type(self):
        result = stress_norm(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
