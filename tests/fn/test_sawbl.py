"""Test sawbl."""
import numpy as np
import pytest
from morie.fn.sawbl import sawbl


def test_sawbl_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawbl(values=vals, n=25)
    assert r.value is not None


def test_sawbl_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawbl(values=vals, n=25)
    assert r.name
