"""Test sawlg."""
import numpy as np
import pytest
from morie.fn.sawlg import sawlg


def test_sawlg_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawlg(values=vals, n=25)
    assert r.value is not None


def test_sawlg_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = sawlg(values=vals, n=25)
    assert r.name
