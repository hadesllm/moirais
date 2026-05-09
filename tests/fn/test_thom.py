"""Tests for moirais.fn.thom — torus homology."""

import pytest

from moirais.fn.thom import torus_homology


class TestTorusHomology:
    def test_2_torus_betti(self):
        r = torus_homology(dim=2)
        assert r.extra["betti_numbers"] == [1, 2, 1]
        assert r.extra["euler_characteristic"] == 0

    def test_3_torus_betti(self):
        r = torus_homology(dim=3)
        assert r.extra["betti_numbers"] == [1, 3, 3, 1]

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_homology(dim=0)
