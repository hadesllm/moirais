"""Tests for moirais.fn.srsiz."""
import numpy as np
from moirais.fn.srsiz import srsiz


def test_srsiz_smoke():
    rng = np.random.default_rng(42)
    result = srsiz(p=0.5)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.srsiz import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
