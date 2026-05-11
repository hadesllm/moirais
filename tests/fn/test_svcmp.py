"""Tests for morie.fn.svcmp -- Committee median voter model"""

import numpy as np
import pytest

from morie.fn.svcmp import committee_med


class TestCommitteeMed:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = committee_med(data)
        assert result.value is not None

    def test_output_type(self):
        result = committee_med(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
