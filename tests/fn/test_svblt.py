"""Tests for moirais.fn.svblt -- Boltzmann (softmax) spatial voting"""

import numpy as np
import pytest

from moirais.fn.svblt import boltzmann_vote


class TestBoltzmannVote:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = boltzmann_vote(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = boltzmann_vote(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
