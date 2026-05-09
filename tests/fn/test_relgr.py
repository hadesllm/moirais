"""Tests for moirais.fn.relgr."""
import numpy as np
from moirais.fn.relgr import relgr


def test_relgr_smoke():
    rng = np.random.default_rng(42)
    result = relgr(failures=np.sort(rng.uniform(0, 100, size=10)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.relgr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
