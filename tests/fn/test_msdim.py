"""Tests for moirais.fn.msdim -- MDS dimensionality selection"""

import numpy as np
import pytest

from moirais.fn.msdim import mds_dims


class TestMdsDims:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_dims(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_dims(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
