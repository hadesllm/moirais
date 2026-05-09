"""Tests for moirais.fn.eigvl -- extract eigenvalues."""

import numpy as np
from moirais.fn.eigvl import extract_eigenvalues, eigvl


def test_eigvl_smoke():
    B = np.diag([5.0, 3.0, 1.0])
    r = eigvl(B)
    assert r.name == "extract_eigenvalues"
    vals = r.value
    assert vals[0] >= vals[1] >= vals[2]
    assert np.isclose(vals[0], 5.0)


def test_eigvl_alias():
    assert eigvl is extract_eigenvalues
