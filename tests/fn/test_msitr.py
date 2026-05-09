"""Tests for moirais.fn.msitr -- MDS iteration convergence"""

import numpy as np
import pytest

from moirais.fn.msitr import mds_iter


class TestMdsIter:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_iter(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_iter(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
