"""Tests for moirais.fn.rcprd -- predicted choice."""

import numpy as np
from moirais.fn.rcprd import predicted_choice, rcprd


def test_rcprd_smoke():
    ideal = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    yea = np.array([0, 0], dtype=float)
    nay = np.array([3, 3], dtype=float)
    r = rcprd(ideal, yea, nay)
    assert r.name == "predicted_choice"
    assert r.value[0] == 1
    assert r.value[2] == 0


def test_rcprd_alias():
    assert rcprd is predicted_choice
