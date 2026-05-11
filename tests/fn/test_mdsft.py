"""Tests for morie.fn.mdsft — MDS fit statistics."""
import numpy as np
import pytest

from morie.fn.mdsft import mdsft


def test_mdsft_smoke():
    eigs = np.array([5.0, 2.0, 0.5, 0.1])
    r = mdsft(eigs)
    assert "fit_by_dim" in r.extra


def test_cheatsheet():
    from morie.fn.mdsft import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
