"""Tests for morie.fn.svwvt -- Weighted voting game value"""

import numpy as np
import pytest

from morie.fn.svwvt import weighted_vote


class TestWeightedVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = weighted_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = weighted_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
