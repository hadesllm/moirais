"""Tests for morie.fn.nomut — NOMINATE utility."""
import numpy as np
import pytest

from morie.fn.nomut import nomut


def test_nomut_smoke():
    x = np.array([[0.5], [-0.5]])
    zy = np.array([[0.3]])
    zn = np.array([[-0.3]])
    r = nomut(x, zy, zn)
    assert "vote_probs" in r.extra


def test_cheatsheet():
    from morie.fn.nomut import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
