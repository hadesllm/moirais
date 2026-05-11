"""Tests for morie.fn.mspro -- Orthogonal Procrustes rotation"""

import numpy as np
import pytest

from morie.fn.mspro import procrustes_orth


class TestProcrustesOrth:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_orth(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_orth(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
