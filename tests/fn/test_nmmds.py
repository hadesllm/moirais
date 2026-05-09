"""Test nmmds."""
import numpy as np
from moirais.fn.nmmds import nonmetric_mds


def test_nmmds_basic():
    r = nonmetric_mds()
    assert r.value is not None


def test_nmmds_name():
    r = nonmetric_mds()
    assert r.name
