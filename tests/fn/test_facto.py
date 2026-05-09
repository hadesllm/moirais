"""Tests for moirais.fn.facto."""
import numpy as np
from moirais.fn.facto import facto


def test_facto_smoke():
    rng = np.random.default_rng(42)
    result = facto()
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.facto import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
