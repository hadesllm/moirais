"""Test saggt."""
import numpy as np
import pytest
from morie.fn.saggt import saggt


def test_saggt_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saggt(values=vals, n=25)
    assert r.value is not None


def test_saggt_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = saggt(values=vals, n=25)
    assert r.name
