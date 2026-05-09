"""Tests for moirais.fn.tfund — fundamental group."""

import pytest

from moirais.fn.tfund import torus_fundamental_group


class TestTorusFundamentalGroup:
    def test_2_torus(self):
        r = torus_fundamental_group(dim=2)
        assert r.extra["group"] == "Z^2"
        assert r.extra["rank"] == 2
        assert len(r.extra["generators"]) == 2

    def test_1_torus(self):
        r = torus_fundamental_group(dim=1)
        assert r.extra["group"] == "Z^1"
        assert r.extra["relations"] == []

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_fundamental_group(dim=0)
