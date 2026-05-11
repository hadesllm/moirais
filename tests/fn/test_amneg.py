"""Tests for morie.fn.amneg — A-M negative weights."""
import numpy as np
from morie.fn.amneg import amneg


def test_amneg_smoke():
    r = amneg([1.0, -0.5, 0.8, -1.2])
    assert r.name == "am_negative_weights"
    assert r.value == 2
    assert r.extra["negative_indices"] == [1, 3]


def test_cheatsheet():
    from morie.fn.amneg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
