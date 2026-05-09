"""Tests for moirais.fn.svlgv -- Logit spatial voting probability"""

import numpy as np
import pytest

from moirais.fn.svlgv import logit_vote


class TestLogitVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = logit_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = logit_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
