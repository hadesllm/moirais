"""Tests for bayes_theorem."""
import pytest
from morie.fn.bayes import bayes_theorem, bayes


def test_basic():
    r = bayes_theorem(0.01, 0.9, evidence=0.05)
    assert abs(r.estimate - 0.18) < 1e-10


def test_binary_complement():
    r = bayes_theorem(0.5, 0.8, likelihood_h0=0.2)
    assert abs(r.estimate - 0.8) < 1e-10


def test_alias():
    assert bayes is bayes_theorem


def test_bad_prior():
    with pytest.raises(ValueError):
        bayes_theorem(-0.1, 0.5, evidence=0.5)


def test_zero_evidence():
    with pytest.raises(ValueError):
        bayes_theorem(0.5, 0.5, evidence=0.0)


def test_no_evidence_no_h0():
    with pytest.raises(ValueError):
        bayes_theorem(0.5, 0.5)
