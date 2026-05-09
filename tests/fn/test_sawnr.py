"""Test sawnr."""
import numpy as np
import pytest
from moirais.fn.sawnr import sawnr


def test_sawnr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnr(values=vals, n=25)
    assert r.value is not None


def test_sawnr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnr(values=vals, n=25)
    assert r.name
