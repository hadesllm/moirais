"""Test saamo."""
import numpy as np
import pytest
from morie.fn.saamo import saamo


def test_saamo_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saamo(values=vals, n=25)
    assert r.value is not None


def test_saamo_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saamo(values=vals, n=25)
    assert r.name
