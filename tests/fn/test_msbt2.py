"""Tests for moirais.fn.msbt2 -- MDS bootstrap confidence"""

import numpy as np
import pytest

from moirais.fn.msbt2 import mds_bootstrap


class TestMdsBootstrap:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_bootstrap(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_bootstrap(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
