"""Tests for morie.fn.svnrm -- Normal kernel vote probability"""

import numpy as np
import pytest

from morie.fn.svnrm import normal_vote


class TestNormalVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = normal_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = normal_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
