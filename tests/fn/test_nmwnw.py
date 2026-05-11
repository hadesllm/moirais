"""Tests for morie.fn.nmwnw -- W-NOMINATE dimension weights"""

import numpy as np
import pytest

from morie.fn.nmwnw import wnominate_weight


class TestWnominateWeight:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wnominate_weight(data)
        assert result.value is not None

    def test_output_type(self):
        result = wnominate_weight(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
