"""Tests for moirais.fn.msst1 -- Raw stress (Kruskal stress-1)"""

import numpy as np
import pytest

from moirais.fn.msst1 import stress_raw


class TestStressRaw:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = stress_raw(X)
        assert result.value is not None

    def test_output_type(self):
        result = stress_raw(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
