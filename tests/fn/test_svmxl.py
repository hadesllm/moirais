"""Tests for moirais.fn.svmxl -- Mixed logit spatial vote"""

import numpy as np
import pytest

from moirais.fn.svmxl import mixed_logit_vote


class TestMixedLogitVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = mixed_logit_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = mixed_logit_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
