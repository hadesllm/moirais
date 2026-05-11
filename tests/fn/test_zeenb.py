"""Tests for morie.fn.zeenb -- Ecological NB regression"""

import numpy as np
import pytest

from morie.fn.zeenb import ecological_nb


class TestEcologicalNb:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ecological_nb(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ecological_nb(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
