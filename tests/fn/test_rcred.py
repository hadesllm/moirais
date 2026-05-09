"""Tests for moirais.fn.rcred -- read roll-call vote matrix."""
import numpy as np
from moirais.fn.rcred import read_roll_call, rcred


def test_alias():
    assert rcred is read_roll_call


def test_smoke():
    votes = np.array([[1, 0, 1], [0, 1, 0], [1, 1, np.nan]])
    r = read_roll_call(votes)
    assert r.name == "read_roll_call"
    assert "yea_count" in r.extra
    assert "nay_count" in r.extra
    assert r.extra["n_legislators"] == 3
    assert r.extra["n_votes"] == 3


def test_missing_count():
    votes = np.array([[1, np.nan], [np.nan, 0]])
    r = read_roll_call(votes)
    assert r.extra["missing"] == 2


def test_yea_nay_counts():
    votes = np.array([[1, 1, 0], [0, 0, 1]])
    r = read_roll_call(votes)
    assert r.extra["yea_count"] == 3
    assert r.extra["nay_count"] == 3
