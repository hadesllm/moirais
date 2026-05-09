"""Tests for moirais.fn.xrjcp -- Join count permutation test"""

import numpy as np
import pytest

from moirais.fn.xrjcp import join_count_perm


class TestJoinCountPerm:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = join_count_perm(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = join_count_perm(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
