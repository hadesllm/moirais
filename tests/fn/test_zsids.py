"""Tests for morie.fn.zsids -- Modified Shepard interpolation"""

import numpy as np
import pytest

from morie.fn.zsids import idw_shepard


class TestIdwShepard:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = idw_shepard(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = idw_shepard(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
