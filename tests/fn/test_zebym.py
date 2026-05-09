"""Tests for moirais.fn.zebym -- BYM (Besag-York-Mollie) model"""

import numpy as np
import pytest

from moirais.fn.zebym import bym_model


class TestBymModel:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bym_model(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = bym_model(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
