"""Tests for moirais.fn.fzand."""
import numpy as np
from moirais.fn.fzand import fzand


def test_fzand_smoke():
    rng = np.random.default_rng(42)
    result = fzand(a=rng.standard_normal(20), b=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fzand import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
