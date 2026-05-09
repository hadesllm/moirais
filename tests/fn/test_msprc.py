"""Tests for moirais.fn.msprc -- Procrustes correlation"""

import numpy as np
import pytest

from moirais.fn.msprc import procrustes_corr


class TestProcrustesCorr:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_corr(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_corr(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
