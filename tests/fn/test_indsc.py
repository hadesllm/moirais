"""Test indsc."""
import numpy as np
from morie.fn.indsc import indscal_mds


def test_indsc_basic():
    r = indscal_mds()
    assert r.value is not None


def test_indsc_name():
    r = indscal_mds()
    assert r.name
