"""Test sagjb."""
import numpy as np
import pytest
from morie.fn.sagjb import sagjb


def test_sagjb_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjb(values=vals, n=25)
    assert r.value is not None


def test_sagjb_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjb(values=vals, n=25)
    assert r.name
