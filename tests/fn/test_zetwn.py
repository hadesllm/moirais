"""Tests for morie.fn.zetwn -- Townsend deprivation index"""

import numpy as np
import pytest

from morie.fn.zetwn import townsend_index


class TestTownsendIndex:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = townsend_index(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = townsend_index(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
