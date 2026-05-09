"""Tests for moirais.fn.svchy -- Cauchy kernel spatial voting"""

import numpy as np
import pytest

from moirais.fn.svchy import cauchy_vote


class TestCauchyVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = cauchy_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = cauchy_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
