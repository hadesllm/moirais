"""Tests for moirais.fn.dceof."""
import numpy as np
from moirais.fn.dceof import dceof


def test_dceof_smoke():
    rng = np.random.default_rng(42)
    result = dceof(set_a={1,2,3,4}, set_b={3,4,5,6})
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.dceof import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
