"""Tests for morie.fn.svdrv -- Directional voting model (Rabinowitz-Macdonald)"""

import numpy as np
import pytest

from morie.fn.svdrv import directional_vote


class TestDirectionalVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = directional_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = directional_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
