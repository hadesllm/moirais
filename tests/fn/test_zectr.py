"""Tests for morie.fn.zectr -- Spatial contact tracing"""

import numpy as np
import pytest

from morie.fn.zectr import contact_trace_sp


class TestContactTraceSp:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = contact_trace_sp(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = contact_trace_sp(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
