"""Test sawtf."""
import numpy as np
import pytest
from morie.fn.sawtf import sawtf


def test_sawtf_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawtf(values=vals, n=25)
    assert r.value is not None


def test_sawtf_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawtf(values=vals, n=25)
    assert r.name
