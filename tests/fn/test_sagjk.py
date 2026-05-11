"""Test sagjk."""
import numpy as np
import pytest
from morie.fn.sagjk import sagjk


def test_sagjk_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjk(values=vals, n=25)
    assert r.value is not None


def test_sagjk_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sagjk(values=vals, n=25)
    assert r.name
