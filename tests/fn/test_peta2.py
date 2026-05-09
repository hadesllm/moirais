"""Tests for moirais.fn.peta2 -- Partial eta-squared."""

import pytest
from moirais.fn.peta2 import partial_eta_squared
from moirais.fn._containers import ESRes


class TestPartialEtaSquared:
    def test_known_value(self):
        """peta2 = SS_effect / (SS_effect + SS_error)."""
        result = partial_eta_squared(ss_effect=20.0, ss_error=80.0)
        assert isinstance(result, ESRes)
        assert result.estimate == pytest.approx(0.2, abs=1e-10)

    def test_zero_effect(self):
        """Zero effect gives peta2 = 0."""
        result = partial_eta_squared(ss_effect=0.0, ss_error=100.0)
        assert result.estimate == pytest.approx(0.0)

    def test_all_effect(self):
        """All variance explained gives peta2 = 1."""
        result = partial_eta_squared(ss_effect=50.0, ss_error=0.0)
        assert result.estimate == pytest.approx(1.0)
