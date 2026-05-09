"""Tests for AIC from entropy."""
import pytest
from moirais.fn.aient import aic_entropy, aient


def test_basic():
    r = aic_entropy(log_likelihood=-100.0, k=3)
    assert r.estimate == pytest.approx(206.0, abs=1e-10)


def test_aicc():
    r = aic_entropy(log_likelihood=-100.0, k=3, n=50)
    assert "AICc" in r.extra
    assert r.extra["AICc"] > r.estimate


def test_alias():
    assert aient is aic_entropy
