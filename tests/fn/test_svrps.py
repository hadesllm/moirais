"""Tests for morie.fn.svrps -- Ranked probability score spatial"""

import numpy as np
import pytest

from morie.fn.svrps import rank_prob_score


class TestRankProbScore:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank_prob_score(data)
        assert result.value is not None

    def test_output_type(self):
        result = rank_prob_score(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
