"""Tests for moirais.fn.svht3 -- Three-candidate spatial equilibrium"""

import numpy as np
import pytest

from moirais.fn.svht3 import hotelling_3cand


class TestHotelling3cand:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = hotelling_3cand(data)
        assert result.value is not None

    def test_output_type(self):
        result = hotelling_3cand(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
