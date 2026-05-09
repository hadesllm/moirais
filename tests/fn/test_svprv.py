"""Tests for moirais.fn.svprv -- Probit spatial voting probability"""

import numpy as np
import pytest

from moirais.fn.svprv import probit_vote


class TestProbitVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = probit_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = probit_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
