"""Tests for moirais.fn.msprp -- Partial Procrustes"""

import numpy as np
import pytest

from moirais.fn.msprp import procrustes_part


class TestProcrustesPart:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = procrustes_part(X)
        assert result.value is not None

    def test_output_type(self):
        result = procrustes_part(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
