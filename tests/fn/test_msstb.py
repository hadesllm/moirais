"""Tests for morie.fn.msstb -- Kruskal stress S2"""

import numpy as np
import pytest

from morie.fn.msstb import stress_s2


class TestStressS2:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = stress_s2(X)
        assert result.value is not None

    def test_output_type(self):
        result = stress_s2(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
