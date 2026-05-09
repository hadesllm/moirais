"""Tests for moirais.fn.msnmt -- Nonmetric MDS (Kruskal)"""

import numpy as np
import pytest

from moirais.fn.msnmt import nonmetric_mds


class TestNonmetricMds:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = nonmetric_mds(X)
        assert result.value is not None

    def test_output_type(self):
        result = nonmetric_mds(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
