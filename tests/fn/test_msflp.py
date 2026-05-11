"""Tests for morie.fn.msflp -- MDS configuration flip check"""

import numpy as np
import pytest

from morie.fn.msflp import mds_flip


class TestMdsFlip:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_flip(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_flip(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
