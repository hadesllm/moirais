"""Tests for moirais.fn.nmovl -- Party overlap index"""

import numpy as np
import pytest

from moirais.fn.nmovl import party_overlap


class TestPartyOverlap:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_overlap(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_overlap(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
