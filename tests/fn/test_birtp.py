"""Tests for morie.fn.birtp — Bayesian IRT posterior summary."""
import numpy as np
import pytest

from morie.fn.birtp import birtp


def test_birtp_smoke():
    chain = np.random.default_rng(42).standard_normal((50, 5, 1))
    r = birtp(chain)
    assert "posterior_mean" in r.extra


def test_cheatsheet():
    from morie.fn.birtp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
