"""Tests for minimum description length."""
import pytest
from morie.fn.mdlen import mdl, mdlen


def test_basic():
    r = mdl(log_likelihood=-100.0, k=3, n=100)
    assert r.estimate > 0


def test_bic_relation():
    r = mdl(log_likelihood=-100.0, k=3, n=100)
    assert r.extra["BIC"] == pytest.approx(2 * r.estimate, rel=1e-10)


def test_alias():
    assert mdlen is mdl
