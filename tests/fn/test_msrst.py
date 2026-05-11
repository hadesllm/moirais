"""Tests for morie.fn.msrst -- R-stress measure"""

import numpy as np
import pytest

from morie.fn.msrst import rstress


class TestRstress:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = rstress(X)
        assert result.value is not None

    def test_output_type(self):
        result = rstress(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
