"""Tests for moirais.fn.birtp — Bayesian IRT posterior summary."""
import numpy as np
import pytest

from moirais.fn.birtp import birtp


def test_birtp_smoke():
    chain = np.random.default_rng(42).standard_normal((50, 5, 1))
    r = birtp(chain)
    assert "posterior_mean" in r.extra


def test_cheatsheet():
    from moirais.fn.birtp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
