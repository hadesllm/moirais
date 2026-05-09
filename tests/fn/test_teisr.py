"""Tests for moirais.fn.teisr — Eisenstein series."""

import pytest

from moirais.fn.teisr import eisenstein_series


class TestEisensteinSeries:
    def test_e4_at_i(self):
        r = eisenstein_series(k=4, tau=1j, terms=30)
        assert isinstance(r.value, float)

    def test_odd_weight_fails(self):
        with pytest.raises(ValueError):
            eisenstein_series(k=3)

    def test_invalid_tau(self):
        with pytest.raises(ValueError):
            eisenstein_series(k=4, tau=1.0)

    def test_unsupported_k(self):
        with pytest.raises(ValueError):
            eisenstein_series(k=16)
