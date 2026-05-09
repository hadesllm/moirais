"""Tests for moirais.fn.msprr -- Procrustes residuals"""

import numpy as np
import pytest

from moirais.fn.msprr import procrustes_resid


class TestProcrustesResid:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_resid(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_resid(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
