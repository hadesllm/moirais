"""Tests for moirais.fn.dblcn — Double centering."""
import numpy as np
import pytest

from moirais.fn.dblcn import dblcn


def test_dblcn_smoke():
    D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2.5], [2, 1.5, 0, 1], [3, 2.5, 1, 0]])
    r = dblcn(D)
    assert r.name == "double_centering"
    assert r.extra["matrix"].shape == (4, 4)


def test_cheatsheet():
    from moirais.fn.dblcn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
