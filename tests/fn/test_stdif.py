"""Tests for moirais.fn.stdif."""
import numpy as np
from moirais.fn.stdif import spatial_did


def test_stdif_smoke():
    rng = np.random.default_rng(42)
    result = spatial_did(
        y=rng.uniform(-80, -75, size=20),
        treated=rng.integers(0, 2, size=20).astype(float),
        post=rng.integers(0, 2, size=20).astype(float),
        coords=rng.uniform(size=(20, 2))
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.stdif import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
