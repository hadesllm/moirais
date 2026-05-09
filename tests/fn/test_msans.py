"""Tests for moirais.fn.msans -- MDS anisotropy measure"""

import numpy as np
import pytest

from moirais.fn.msans import mds_aniso


class TestMdsAniso:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_aniso(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_aniso(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
