"""Tests for moirais.fn.oocls — Optimal classification."""
import numpy as np
import pytest

from moirais.fn.oocls import oocls


def test_oocls_smoke():
    votes = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0]], dtype=float)
    r = oocls(votes)
    assert r.name == "optimal_classification"
    assert 0 <= r.value <= 1


def test_cheatsheet():
    from moirais.fn.oocls import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
