"""Tests for second-order intensity."""
import numpy as np
from morie.fn.sg2nd import sg2nd


def test_sg2nd_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (40, 2))
    r = sg2nd(pts, (0, 10, 0, 10))
    assert r.name == "second_order_intensity"
    assert "lambda2_values" in r.extra
    assert r.extra["first_order_intensity"] > 0


def test_cheatsheet():
    from morie.fn.sg2nd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
