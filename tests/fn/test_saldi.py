"""Test saldi."""
import numpy as np
import pytest
from moirais.fn.saldi import saldi


def test_saldi_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saldi(values=vals, n=25)
    assert r.value is not None


def test_saldi_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saldi(values=vals, n=25)
    assert r.name
