"""Test sawnv."""
import numpy as np
import pytest
from morie.fn.sawnv import sawnv


def test_sawnv_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnv(values=vals, n=25)
    assert r.value is not None


def test_sawnv_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnv(values=vals, n=25)
    assert r.name
