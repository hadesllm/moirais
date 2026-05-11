"""Tests for morie.fn.svcwn -- Condorcet winner test"""

import numpy as np
import pytest

from morie.fn.svcwn import condorcet_winner


class TestCondorcetWinner:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = condorcet_winner(data)
        assert result.value is not None

    def test_output_type(self):
        result = condorcet_winner(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
