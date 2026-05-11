"""Tests for morie.fn.svcm2 -- 2D committee decision"""

import numpy as np
import pytest

from morie.fn.svcm2 import committee_2d


class TestCommittee2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = committee_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = committee_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
