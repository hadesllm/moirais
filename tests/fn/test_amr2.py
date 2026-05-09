"""Tests for moirais.fn.amr2 — A-M R-squared."""
import numpy as np
from moirais.fn.amr2 import amr2


def test_amr2_perfect():
    r = amr2([1, 2, 3], [1, 2, 3])
    assert r.name == "am_r_squared"
    assert abs(r.value - 1.0) < 1e-10


def test_amr2_imperfect():
    r = amr2([1, 2, 3], [1.1, 2.2, 2.8])
    assert 0 < r.value < 1
