"""Tests for morie.fn.nmdwn -- DW-NOMINATE dynamic estimation"""

import numpy as np
import pytest

from morie.fn.nmdwn import dwnominate


class TestDwnominate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dwnominate(data)
        assert result.value is not None

    def test_output_type(self):
        result = dwnominate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
