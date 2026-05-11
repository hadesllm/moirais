"""Tests for morie.fn.eigvc -- extract eigenvectors."""

import numpy as np
from morie.fn.eigvc import extract_eigenvectors, eigvc


def test_eigvc_smoke():
    B = np.diag([5.0, 3.0, 1.0])
    r = eigvc(B, n_dims=2)
    assert r.name == "extract_eigenvectors"
    assert r.value.shape == (3, 2)
    assert r.extra["n_dims"] == 2


def test_eigvc_alias():
    assert eigvc is extract_eigenvectors
