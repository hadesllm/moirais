"""Tests for morie.fn.qvote — Quadratic voting."""
import numpy as np
from morie.fn.qvote import qvote


def test_qvote_basic():
    I = np.array([[10, -5, 1], [1, 10, -5]])
    r = qvote(I)
    assert "outcomes" in r.value
    assert len(r.value["outcomes"]) == 3


def test_qvote_unanimous():
    I = np.array([[10, 10], [10, 10]])
    r = qvote(I)
    assert all(r.value["outcomes"] == 1)


def test_qvote_budget():
    I = np.array([[1, -1]])
    r = qvote(I, budget=50.0)
    assert r.extra["budget"] == 50.0
