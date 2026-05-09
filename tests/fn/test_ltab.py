"""Tests for moirais.fn.ltab -- Abridged life table."""

import numpy as np
import pytest
from moirais.fn.ltab import life_table


class TestLifeTable:
    def test_lx_starts_at_radix(self):
        result = life_table([0, 5, 10], [100, 50, 200], [10000, 9000, 8000], radix=100000)
        assert result["lx"][0] == 100000

    def test_lx_monotone_decreasing(self):
        result = life_table([0, 1, 5, 10], [50, 20, 30, 100], [5000, 4000, 3500, 3000])
        lx = result["lx"]
        for i in range(len(lx) - 1):
            assert lx[i] >= lx[i + 1]

    def test_last_qx_is_one(self):
        result = life_table([0, 10, 20], [10, 20, 50], [1000, 800, 500])
        assert result["nqx"][-1] == 1.0

    def test_ex_positive(self):
        result = life_table([0, 5, 10, 15], [5, 3, 4, 10], [1000, 900, 800, 600])
        assert all(e > 0 for e in result["ex"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            life_table([0, 5], [10], [100, 90])
