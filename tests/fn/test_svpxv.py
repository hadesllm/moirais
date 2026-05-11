"""Tests for morie.fn.svpxv -- Proximity voting model probability"""

import numpy as np
import pytest

from morie.fn.svpxv import proximity_vote


class TestProximityVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = proximity_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = proximity_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
