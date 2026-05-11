"""Tests for morie.fn.smcuf — SMACOF unfolding."""
import numpy as np
import pytest

from morie.fn.smcuf import smcuf


def test_smcuf_smoke():
    D_u = np.random.default_rng(42).random((4, 3)) + 0.5
    r = smcuf(D_u, n_dims=1, max_iter=20)
    assert "respondent_coords" in r.extra


def test_cheatsheet():
    from morie.fn.smcuf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
