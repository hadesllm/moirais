"""Tests for morie.fn.mlsms -- MLSMU6 missing impute."""

import numpy as np
from morie.fn.mlsms import mlsmu6_missing_impute, mlsms


def test_mlsms_mean():
    D = np.array([[1, np.nan, 3], [np.nan, 2, 4]], dtype=float)
    r = mlsms(D, method="mean")
    assert r.name == "mlsmu6_missing_impute"
    assert not np.any(np.isnan(r.value))
    assert r.extra["n_imputed"] == 2


def test_mlsms_global():
    D = np.array([[1, np.nan], [2, 3]], dtype=float)
    r = mlsms(D, method="global_mean")
    assert np.isclose(r.value[0, 1], 2.0)


def test_mlsms_alias():
    assert mlsms is mlsmu6_missing_impute
