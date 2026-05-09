"""Tests for moirais.fn.voron."""
import numpy as np
from moirais.fn.voron import voron


def test_voron_smoke():
    rng = np.random.default_rng(42)
    result = voron(points=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.voron import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
