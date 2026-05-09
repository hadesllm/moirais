"""Tests for moirais.fn.msjck -- MDS jackknife stability"""

import numpy as np
import pytest

from moirais.fn.msjck import mds_jackknife


class TestMdsJackknife:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_jackknife(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_jackknife(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
