"""Tests for morie.fn.claim."""
import numpy as np
from morie.fn.claim import claim


def test_claim_smoke():
    rng = np.random.default_rng(42)
    result = claim(
        n_claims=rng.integers(0, 10, size=20).astype(float),
        exposure=rng.uniform(100, 1000, size=20)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.claim import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
