"""Tests for moirais.fn.mseig -- MDS eigendecomposition"""

import numpy as np
import pytest

from moirais.fn.mseig import mds_eigen


class TestMdsEigen:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_eigen(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_eigen(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
