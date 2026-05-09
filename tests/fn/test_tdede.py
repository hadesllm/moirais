"""Tests for moirais.fn.tdede — Dedekind eta function."""

import pytest

from moirais.fn.tdede import dedekind_eta


class TestDedekindEta:
    def test_positive_abs(self):
        r = dedekind_eta(tau=1j)
        assert r.extra["abs"] > 0

    def test_returns_float(self):
        r = dedekind_eta()
        assert isinstance(r.value, float)

    def test_invalid(self):
        with pytest.raises(ValueError):
            dedekind_eta(tau=1.0)
