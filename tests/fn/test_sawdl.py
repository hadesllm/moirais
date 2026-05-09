"""Test sawdl."""
import numpy as np
import pytest
from moirais.fn.sawdl import sawdl


def test_sawdl_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawdl(values=vals, n=25)
    assert r.value is not None


def test_sawdl_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawdl(values=vals, n=25)
    assert r.name
