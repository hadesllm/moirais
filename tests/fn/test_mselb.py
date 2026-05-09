"""Tests for moirais.fn.mselb -- MDS elbow detection"""

import numpy as np
import pytest

from moirais.fn.mselb import mds_elbow


class TestMdsElbow:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_elbow(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_elbow(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
