"""Tests for moirais.fn.impme -- impute missing mean."""

import numpy as np
from moirais.fn.impme import impute_missing_mean, impme


def test_impme_no_missing():
    D = np.ones((3, 3))
    r = impme(D)
    assert r.name == "impute_missing_mean"
    assert r.extra["n_imputed"] == 0


def test_impme_with_nan():
    D = np.array([[0, 1, np.nan], [1, 0, 2], [np.nan, 2, 0]])
    r = impme(D)
    assert not np.any(np.isnan(r.value))
    assert r.extra["n_imputed"] == 2


def test_impme_alias():
    assert impme is impute_missing_mean
