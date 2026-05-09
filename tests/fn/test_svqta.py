"""Tests for moirais.fn.svqta -- Quota game equilibrium"""

import numpy as np
import pytest

from moirais.fn.svqta import quota_game


class TestQuotaGame:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = quota_game(data)
        assert result.value is not None

    def test_output_type(self):
        result = quota_game(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
