"""Tests for morie.fn.edtds."""
import numpy as np
from morie.fn.edtds import edtds


def test_edtds_smoke():
    rng = np.random.default_rng(42)
    result = edtds(str_a="kitten", str_b="sitting")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.edtds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
