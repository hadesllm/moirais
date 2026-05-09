"""Tests for moirais.fn.msplt -- MDS polarity detection"""

import numpy as np
import pytest

from moirais.fn.msplt import mds_polarity


class TestMdsPolarity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_polarity(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_polarity(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
