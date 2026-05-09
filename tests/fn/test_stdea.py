"""Test stdea."""
import numpy as np
from moirais.fn.stdea import st_diff_equation


def test_stdea_basic():
    r = st_diff_equation()
    assert r.statistic is not None


def test_stdea_name():
    r = st_diff_equation()
    assert r.name
