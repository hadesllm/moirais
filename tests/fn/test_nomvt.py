"""Tests for moirais.fn.nomvt — NOMINATE vote probability."""
import numpy as np
import pytest

from moirais.fn.nomvt import nomvt


def test_nomvt_smoke():
    r = nomvt(np.array([0.5]), np.array([0.3]), np.array([-0.3]))
    assert 0 <= r.value <= 1


def test_cheatsheet():
    from moirais.fn.nomvt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
