"""Tests for morie.fn.svpp2 -- Party position 2D (Laver-Hunt)"""

import numpy as np
import pytest

from morie.fn.svpp2 import party_pos_2d


class TestPartyPos2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = party_pos_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = party_pos_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
