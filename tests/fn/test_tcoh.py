"""Tests for morie.fn.tcoh — torus cohomology."""

import pytest

from morie.fn.tcoh import torus_cohomology


class TestTorusCohomology:
    def test_2_torus(self):
        r = torus_cohomology(dim=2)
        assert r.extra["betti_numbers"] == [1, 2, 1]
        assert r.extra["total_betti"] == 4

    def test_1_torus(self):
        r = torus_cohomology(dim=1)
        assert r.extra["betti_numbers"] == [1, 1]

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_cohomology(dim=0)
