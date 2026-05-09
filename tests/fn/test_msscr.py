"""Tests for moirais.fn.msscr -- MDS scree plot values"""

import numpy as np
import pytest

from moirais.fn.msscr import mds_scree


class TestMdsScree:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = mds_scree(X)
        assert result.value is not None

    def test_output_type(self):
        result = mds_scree(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
