"""Tests for moirais.fn.ocbin — Optimal classification binarized."""
import numpy as np
import pytest

from moirais.fn.ocbin import ocbin


def test_ocbin_smoke():
    ratings = np.array([[5, 3, 1], [4, 4, 2], [2, 3, 5], [1, 5, 3], [3, 2, 4]], dtype=float)
    r = ocbin(ratings)
    assert "PRE" in r.extra


def test_cheatsheet():
    from moirais.fn.ocbin import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
