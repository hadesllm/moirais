"""Tests for moirais.fn.latns."""
import numpy as np
from moirais.fn.latns import latns


def test_latns_smoke():
    rng = np.random.default_rng(42)
    result = latns(n_samples=50, n_dims=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.latns import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
