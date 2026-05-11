"""Tests for morie.fn.svclr -- Condorcet loser identification"""

import numpy as np
import pytest

from morie.fn.svclr import condorcet_loser


class TestCondorcetLoser:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = condorcet_loser(data)
        assert result.value is not None

    def test_output_type(self):
        result = condorcet_loser(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
