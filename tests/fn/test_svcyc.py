"""Tests for moirais.fn.svcyc -- Condorcet cycle detection"""

import numpy as np
import pytest

from moirais.fn.svcyc import condorcet_cycle


class TestCondorcetCycle:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = condorcet_cycle(data)
        assert result.value is not None

    def test_output_type(self):
        result = condorcet_cycle(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
