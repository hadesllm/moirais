"""Tests for morie.fn.polcd — polar code construction."""

import numpy as np
import pytest

from morie.fn.polcd import polcd


class TestPolcd:
    def test_basic_construction(self):
        result = polcd(8, 4)
        assert len(result["info_bits"]) == 4
        assert len(result["frozen_bits"]) == 4

    def test_all_bits_covered(self):
        result = polcd(16, 8)
        all_bits = set(result["info_bits"]) | set(result["frozen_bits"])
        assert all_bits == set(range(16))

    def test_rate(self):
        result = polcd(32, 16)
        assert result["rate"] == pytest.approx(0.5)

    def test_bhattacharyya_length(self):
        result = polcd(8, 4)
        assert len(result["bhattacharyya"]) == 8

    def test_not_power_of_two(self):
        with pytest.raises(ValueError):
            polcd(6, 3)

    def test_k_out_of_range(self):
        with pytest.raises(ValueError):
            polcd(8, 0)
        with pytest.raises(ValueError):
            polcd(8, 9)

    def test_bhattacharyya_between_0_and_1(self):
        result = polcd(8, 4, design_snr_db=0.0)
        assert np.all(result["bhattacharyya"] >= 0)
        assert np.all(result["bhattacharyya"] <= 1.0 + 1e-10)
