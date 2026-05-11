"""Tests for morie.fn.svppo -- Party position estimation"""

import numpy as np
import pytest

from morie.fn.svppo import party_position


class TestPartyPosition:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_position(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_position(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
