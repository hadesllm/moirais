"""Tests for morie.fn.nmwno -- W-NOMINATE estimation"""

import numpy as np
import pytest

from morie.fn.nmwno import wnominate


class TestWnominate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wnominate(data)
        assert result.value is not None

    def test_output_type(self):
        result = wnominate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
