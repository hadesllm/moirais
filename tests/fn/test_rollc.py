"""Tests for morie.fn.rollc — Roll call analysis."""
import numpy as np
from morie.fn.rollc import rollc


def test_rollc_basic():
    V = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    r = rollc(V)
    assert len(r.value["participation_rates"]) == 3
    assert all(p == 1.0 for p in r.value["participation_rates"])


def test_rollc_missing():
    V = np.array([[1, np.nan], [0, 1]], dtype=float)
    r = rollc(V)
    assert r.value["participation_rates"][0] == 0.5


def test_rollc_close():
    V = np.array([[1, 0], [0, 1]], dtype=float)
    r = rollc(V)
    assert r.extra["n_close_votes"] == 2
