"""Tests for standardized prediction error."""
import numpy as np
from moirais.fn.sgkse import sgkse


def test_sgkse_smoke():
    errors = np.array([0.1, -0.2, 0.15, -0.05])
    krig_var = np.array([0.04, 0.05, 0.03, 0.06])
    r = sgkse(errors, krig_var)
    assert r.name == "standardized_prediction_error"
    assert "std_errors" in r.extra
    assert len(r.extra["std_errors"]) == 4


def test_cheatsheet():
    from moirais.fn.sgkse import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
