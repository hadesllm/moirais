"""Tests for moirais.fn.stacf."""
import numpy as np
from moirais.fn.stacf import st_autocorrelation


def test_stacf_smoke():
    rng = np.random.default_rng(42)
    result = st_autocorrelation(
        values=rng.standard_normal(20),
        coords=rng.uniform(size=(20, 2)),
        times=np.arange(20, dtype=float)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.stacf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
