"""Tests for moirais.fn.stkde."""
import numpy as np
from moirais.fn.stkde import st_kde


def test_stkde_smoke():
    rng = np.random.default_rng(42)
    result = st_kde(coords=rng.uniform(size=(20, 2)), times=np.arange(20, dtype=float))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.stkde import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
