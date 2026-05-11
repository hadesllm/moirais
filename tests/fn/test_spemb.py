"""Tests for morie.fn.spemb."""
import numpy as np
from morie.fn.spemb import spemb


def test_spemb_smoke():
    rng = np.random.default_rng(42)
    result = spemb(X=rng.standard_normal((30, 3)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.spemb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
