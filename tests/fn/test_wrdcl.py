"""Tests for morie.fn.wrdcl."""
import numpy as np
from morie.fn.wrdcl import word_cloud_data


def test_wrdcl_smoke():
    rng = np.random.default_rng(42)
    result = word_cloud_data(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.wrdcl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
