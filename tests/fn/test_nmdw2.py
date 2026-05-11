"""Tests for morie.fn.nmdw2 -- DW-NOMINATE bridging observations"""

import numpy as np
import pytest

from morie.fn.nmdw2 import dwnominate_bridge


class TestDwnominateBridge:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dwnominate_bridge(data)
        assert result.value is not None

    def test_output_type(self):
        result = dwnominate_bridge(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
