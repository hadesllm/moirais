"""Tests for moirais.fn.mssmc -- SMACOF iterative MDS"""

import numpy as np
import pytest

from moirais.fn.mssmc import smacof_mds


class TestSmacofMds:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        result = smacof_mds(X)
        assert result.value is not None

    def test_output_type(self):
        result = smacof_mds(np.random.default_rng(0).standard_normal((5,2)))
        assert hasattr(result, "value")
