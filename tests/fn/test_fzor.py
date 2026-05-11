"""Tests for morie.fn.fzor."""
import numpy as np
from morie.fn.fzor import fzor


def test_fzor_smoke():
    rng = np.random.default_rng(42)
    result = fzor(a=rng.standard_normal(20), b=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.fzor import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
