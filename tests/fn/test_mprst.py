"""Tests for mprst.py - Matching Pursuit decomposition."""
import numpy as np
import pytest
from moirais.fn.mprst import matching_pursuit_fn, mprst


def test_matching_pursuit_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = matching_pursuit_fn(x)
    assert result.name == "matching_pursuit"
    assert isinstance(result.value, (int, float))


def test_matching_pursuit_extra_dict():
    x = np.random.default_rng(42).standard_normal(256)
    result = matching_pursuit_fn(x, n_atoms=5)
    assert isinstance(result.extra, dict)


def test_matching_pursuit_n_atoms_param():
    x = np.random.default_rng(42).standard_normal(256)
    result = matching_pursuit_fn(x, n_atoms=3)
    assert result.value is not None


def test_mprst_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = mprst(x, n_atoms=5)
    assert result.name == "matching_pursuit"
