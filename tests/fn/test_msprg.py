"""Tests for morie.fn.msprg -- Generalized Procrustes analysis"""

import numpy as np
import pytest

from morie.fn.msprg import procrustes_gen


class TestProcrustesGen:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_gen(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_gen(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
