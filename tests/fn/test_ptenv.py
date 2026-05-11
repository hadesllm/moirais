"""Tests for morie.fn.ptenv -- Point pattern Monte Carlo envelope"""

import numpy as np
import pytest

from morie.fn.ptenv import pp_envelope


class TestPpEnvelope:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pp_envelope(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pp_envelope(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
