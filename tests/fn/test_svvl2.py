"""Tests for moirais.fn.svvl2 -- 2D valence spatial model"""

import numpy as np
import pytest

from moirais.fn.svvl2 import valence_2d


class TestValence2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = valence_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = valence_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
