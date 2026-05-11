"""Tests for morie.fn.nmf."""
import numpy as np
from morie.fn.nmf import nmf


def test_nmf_smoke():
    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal((30, 10))) + 0.01
    result = nmf(X=X, n_components=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.nmf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
