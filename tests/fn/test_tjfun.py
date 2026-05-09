"""Tests for moirais.fn.tjfun — j-invariant."""

import pytest

from moirais.fn.tjfun import j_invariant


class TestJInvariant:
    def test_tau_i(self):
        r = j_invariant(tau=1j)
        assert r.extra["j_real"] == pytest.approx(1728.0, rel=1e-3)

    def test_returns_real(self):
        r = j_invariant(tau=0.5j + 0.5)
        assert isinstance(r.value, float)

    def test_invalid(self):
        with pytest.raises(ValueError):
            j_invariant(tau=1.0)
