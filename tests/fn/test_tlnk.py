"""Tests for morie.fn.tlnk — torus link."""

import pytest

from morie.fn.tlnk import torus_link


class TestTorusLink:
    def test_knot_single_component(self):
        r = torus_link(p=2, q=3)
        assert r.extra["n_components"] == 1
        assert r.extra["linking_number"] == 0

    def test_hopf_link(self):
        r = torus_link(p=2, q=2)
        assert r.extra["n_components"] == 2
        assert r.extra["linking_number"] == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_link(p=0, q=3)
