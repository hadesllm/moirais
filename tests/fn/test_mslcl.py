"""Tests for morie.fn.mslcl -- Local continuity meta-criterion"""

import numpy as np
import pytest

from morie.fn.mslcl import lcmc


class TestLcmc:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lcmc(data)
        assert result.value is not None

    def test_output_type(self):
        result = lcmc(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
