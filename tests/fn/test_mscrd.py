"""Tests for moirais.fn.mscrd -- MDS coordinate extraction"""

import numpy as np
import pytest

from moirais.fn.mscrd import mds_coords


class TestMdsCoords:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_coords(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_coords(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
