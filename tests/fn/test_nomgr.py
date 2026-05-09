"""Tests for moirais.fn.nomgr."""
import numpy as np
from moirais.fn.nomgr import nomgr


def test_nomgr_smoke():
    rng = np.random.default_rng(42)
    result = nomgr(prior=0.1, lr_pos=4.5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.nomgr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
