"""Tests for morie.fn.zee2s -- Enhanced 2SFCA"""

import numpy as np
import pytest

from morie.fn.zee2s import enhanced_2sfca


class TestEnhanced2sfca:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = enhanced_2sfca(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = enhanced_2sfca(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
