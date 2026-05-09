"""Tests for moirais.fn.mssts -- Kruskal stress S1"""

import numpy as np
import pytest

from moirais.fn.mssts import stress_s1


class TestStressS1:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = stress_s1(X)
        assert result.value is not None

    def test_output_type(self):
        result = stress_s1(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
