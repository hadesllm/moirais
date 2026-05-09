"""Tests for moirais.fn.msswt -- missing data weights."""

import numpy as np
from moirais.fn.msswt import missing_data_weights, msswt


def test_msswt_no_missing():
    D = np.ones((3, 3))
    r = msswt(D)
    assert r.name == "missing_data_weights"
    assert np.all(r.value == 1.0)
    assert r.extra["n_missing"] == 0


def test_msswt_with_nan():
    D = np.array([[0, 1, np.nan], [1, 0, 2], [np.nan, 2, 0]])
    r = msswt(D)
    assert r.extra["n_missing"] == 2
    assert r.value[0, 2] == 0.0


def test_msswt_alias():
    assert msswt is missing_data_weights
