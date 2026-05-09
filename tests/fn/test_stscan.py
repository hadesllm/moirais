"""Tests for moirais.fn.stscan."""
import numpy as np
from moirais.fn.stscan import st_scan_statistic


def test_stscan_smoke():
    rng = np.random.default_rng(42)
    result = st_scan_statistic(
        coords=rng.uniform(size=(20, 2)),
        times=np.arange(20, dtype=float),
        cases=rng.integers(0, 10, size=20).astype(float),
        population=rng.uniform(100, 1000, size=20)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.stscan import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
