"""Tests for moirais.fn.cmds — Classical multidimensional scaling."""
import numpy as np
import pytest

from moirais.fn.cmds import cmds


def test_cmds_smoke():
    D = np.array([[0, 1, 2, 3], [1, 0, 1.5, 2.5], [2, 1.5, 0, 1], [3, 2.5, 1, 0]])
    r = cmds(D)
    assert r.coordinates.shape == (4, 2)
    assert r.stress >= 0


def test_cheatsheet():
    from moirais.fn.cmds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
