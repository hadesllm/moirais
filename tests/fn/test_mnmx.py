"""Tests for morie.fn.mnmx — minimax."""
import numpy as np
import pytest
from morie.fn.mnmx import minimax


class TestMinimax:
    def test_pure_strategy(self):
        A = np.array([[3, -1], [-1, 3]])
        res = minimax(A)
        assert res.value is not None

    def test_saddle_point(self):
        A = np.array([[1, 2], [3, 4]])
        res = minimax(A)
        assert res.extra["game_value"] >= 1
