"""Tests for morie.fn.tmodu — torus modular parameter."""

import pytest

from morie.fn.tmodu import torus_modular


class TestTorusModular:
    def test_standard(self):
        r = torus_modular(tau=1j)
        assert r.extra["area"] == pytest.approx(1.0)
        assert r.extra["in_fundamental_domain"]

    def test_not_fundamental(self):
        r = torus_modular(tau=0.6 + 1j)
        assert not r.extra["in_fundamental_domain"]

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_modular(tau=1.0 + 0j)
