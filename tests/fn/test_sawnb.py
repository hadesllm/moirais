"""Test sawnb."""
import numpy as np
import pytest
from morie.fn.sawnb import sawnb


def test_sawnb_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnb(values=vals, n=25)
    assert r.value is not None


def test_sawnb_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawnb(values=vals, n=25)
    assert r.name
