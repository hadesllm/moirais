"""Tests for morie.fn.bscch — BSC capacity."""

import numpy as np
import pytest

from morie.fn.bscch import bscch


class TestBscch:
    def test_noiseless(self):
        assert bscch(0.0)["capacity"] == pytest.approx(1.0)

    def test_completely_noisy(self):
        assert bscch(1.0)["capacity"] == pytest.approx(1.0)

    def test_half_crossover(self):
        assert bscch(0.5)["capacity"] == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        c1 = bscch(0.1)["capacity"]
        c2 = bscch(0.9)["capacity"]
        assert c1 == pytest.approx(c2, abs=1e-10)

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            bscch(-0.1)
        with pytest.raises(ValueError):
            bscch(1.5)

    def test_known_value(self):
        c = bscch(0.11)["capacity"]
        h = -0.11 * np.log2(0.11) - 0.89 * np.log2(0.89)
        assert c == pytest.approx(1 - h, abs=1e-10)
