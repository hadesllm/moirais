"""Tests for omprs.py - Orthogonal Matching Pursuit decomposition."""
import numpy as np
import pytest
from morie.fn.omprs import omp_fn, omprs


def test_omp_returns_descriptive_result():
    x = np.random.default_rng(42).standard_normal(256)
    result = omp_fn(x)
    assert result.name == "orthogonal_matching_pursuit"
    assert isinstance(result.value, (int, float))


def test_omp_extra_dict():
    x = np.random.default_rng(42).standard_normal(256)
    result = omp_fn(x, n_atoms=5)
    assert isinstance(result.extra, dict)


def test_omp_n_atoms_param():
    x = np.random.default_rng(42).standard_normal(256)
    result = omp_fn(x, n_atoms=3)
    assert result.value is not None


def test_omprs_alias():
    x = np.random.default_rng(42).standard_normal(128)
    result = omprs(x, n_atoms=5)
    assert result.name == "orthogonal_matching_pursuit"
