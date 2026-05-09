"""Tests for moirais.fn.rcdrp -- drop influential votes."""

import numpy as np
from moirais.fn.rcdrp import drop_influential_votes, rcdrp


def test_rcdrp_keep_all():
    votes = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=float)
    r = rcdrp(votes, threshold=0.9)
    assert r.name == "drop_influential_votes"
    assert r.value.shape[1] == 3


def test_rcdrp_drop():
    votes = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
    r = rcdrp(votes, threshold=0.8)
    assert r.extra["n_dropped"] >= 1


def test_rcdrp_alias():
    assert rcdrp is drop_influential_votes
