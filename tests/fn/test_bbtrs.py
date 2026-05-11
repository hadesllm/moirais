"""Test bbtrs."""
import numpy as np
from morie.fn.bbtrs import blackbox_transpose


def test_bbtrs_basic():
    r = blackbox_transpose()
    assert r.value is not None


def test_bbtrs_name():
    r = blackbox_transpose()
    assert r.name
