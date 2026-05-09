"""Tests for icafn.py - Independent Component Analysis."""
import numpy as np
import pytest
from moirais.fn.icafn import ica_fn, icafn


def test_ica_returns_descriptive_result():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 256))
    result = ica_fn(X)
    assert result.name == "ica"
    assert isinstance(result.value, (int, np.integer))
    assert "sources" in result.extra
    assert "mixing_matrix" in result.extra


def test_ica_source_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 256))
    result = ica_fn(X, n_components=2)
    assert result.extra["sources"].shape[0] == 2
    assert result.extra["sources"].shape[1] == 256


def test_ica_mixing_matrix_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 256))
    result = ica_fn(X, n_components=2)
    assert result.extra["mixing_matrix"].shape == (3, 2)


def test_icafn_alias():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((2, 128))
    result = icafn(X)
    assert result.name == "ica"
