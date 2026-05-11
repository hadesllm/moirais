"""Tests for morie.fn.zeby2 -- BYM2 reparameterized model"""

import numpy as np
import pytest

from morie.fn.zeby2 import bym2_model


class TestBym2Model:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bym2_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = bym2_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
