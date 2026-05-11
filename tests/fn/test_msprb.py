"""Tests for morie.fn.msprb -- Oblique Procrustes rotation"""

import numpy as np
import pytest

from morie.fn.msprb import procrustes_obl


class TestProcrustesObl:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_obl(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_obl(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
