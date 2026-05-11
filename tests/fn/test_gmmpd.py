"""Tests for gmm_pdf."""
import numpy as np
import pytest
from morie.fn.gmmpd import gmm_pdf, gmmpd


def test_single_component():
    x = np.array([0.0])
    r = gmm_pdf(x, means=[0], covs=[1], weights=[1])
    expected = 1.0 / np.sqrt(2 * np.pi)
    assert abs(r.extra["pdf"][0] - expected) < 1e-6


def test_alias():
    assert gmmpd is gmm_pdf


def test_bad_variance():
    with pytest.raises(ValueError):
        gmm_pdf([0], [0], [0], [1])


def test_two_components():
    x = np.linspace(-3, 3, 100)
    r = gmm_pdf(x, means=[-1, 1], covs=[0.5, 0.5], weights=[0.5, 0.5])
    assert r.extra["n_components"] == 2
