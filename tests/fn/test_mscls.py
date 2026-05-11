"""Tests for morie.fn.mscls -- Classical MDS (Torgerson)"""

import numpy as np
import pytest

from morie.fn.mscls import classical_mds


class TestClassicalMds:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = classical_mds(X)
        assert result.value is not None

    def test_output_type(self):
        result = classical_mds(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
