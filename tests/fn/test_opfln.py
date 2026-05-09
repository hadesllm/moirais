"""Test opfln."""
import numpy as np
import pytest
from moirais.fn.opfln import opfln


def test_opfln_basic():
    rng = np.random.default_rng(42)
    r = opfln(n_dims=2, max_iter=50)
    assert r.value is not None


def test_opfln_description():
    rng = np.random.default_rng(42)
    r = opfln(n_dims=2, max_iter=50)
    assert r.name
