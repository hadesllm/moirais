"""Tests for moirais.fn.msrsq -- MDS R-squared goodness of fit"""

import numpy as np
import pytest

from moirais.fn.msrsq import mds_rsq


class TestMdsRsq:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_rsq(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_rsq(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
