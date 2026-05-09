"""Tests for moirais.fn.nmwn2 -- W-NOMINATE 2D"""

import numpy as np
import pytest

from moirais.fn.nmwn2 import wnominate_2d


class TestWnominate2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wnominate_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = wnominate_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
