"""Tests for moirais.fn.vgpsr -- Partial sill ratio"""

import numpy as np
import pytest

from moirais.fn.vgpsr import partial_sill_ratio


class TestPartialSillRatio:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = partial_sill_ratio(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = partial_sill_ratio(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
