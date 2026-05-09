"""Tests for moirais.fn.tweir — Weierstrass p-function."""

import pytest

from moirais.fn.tweir import weierstrass_p


class TestWeierstrassP:
    def test_symmetry(self):
        r1 = weierstrass_p(z=0.3 + 0.2j, tau=1j, terms=5)
        r2 = weierstrass_p(z=-0.3 - 0.2j, tau=1j, terms=5)
        assert r1.extra["real"] == pytest.approx(r2.extra["real"], rel=1e-3)

    def test_returns_float(self):
        r = weierstrass_p()
        assert isinstance(r.value, float)

    def test_invalid(self):
        with pytest.raises(ValueError):
            weierstrass_p(tau=1.0)
