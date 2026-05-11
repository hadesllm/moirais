"""Tests for morie.fn.spectc."""
import numpy as np
from morie.fn.spectc import spectc


def test_spectc_smoke():
    rng = np.random.default_rng(42)
    result = spectc(affinity=np.exp(-rng.uniform(size=(20, 20))))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.spectc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
