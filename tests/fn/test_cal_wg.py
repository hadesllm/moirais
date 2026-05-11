"""Tests for morie.fn.cal_wg — Raking calibration weights."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.cal_wg import calibration_weights


@pytest.fixture()
def cal_data():
    """DataFrame with two numeric auxiliary variables."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age": rng.uniform(20, 70, size=n),
        "income": rng.uniform(20000, 100000, size=n),
        "y": rng.standard_normal(n),
    })


def test_returns_series(cal_data):
    """calibration_weights returns a pd.Series of correct length."""
    pop_totals = {"age": 9000.0, "income": 12_000_000.0}
    result = calibration_weights(cal_data, ["age", "income"], pop_totals)
    assert isinstance(result, pd.Series)
    assert len(result) == len(cal_data)


def test_weights_positive(cal_data):
    """All calibration weights should be positive."""
    pop_totals = {"age": 9000.0, "income": 12_000_000.0}
    result = calibration_weights(cal_data, ["age", "income"], pop_totals)
    assert (result > 0).all()


def test_calibrated_totals_match_targets():
    """Weighted totals should match the population target with single variable."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"age": rng.uniform(20, 70, size=n)})
    # Single-variable raking converges exactly
    pop_totals = {"age": 9000.0}
    w = calibration_weights(df, ["age"], pop_totals, max_iter=100)
    weighted_total = (w.values * df["age"].values).sum()
    assert weighted_total == pytest.approx(9000.0, rel=1e-4)


def test_missing_var_raises(cal_data):
    """Auxiliary variable not in DataFrame should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        calibration_weights(cal_data, ["nonexistent"], {"nonexistent": 100.0})


def test_missing_pop_total_raises(cal_data):
    """Auxiliary variable not in population_totals should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        calibration_weights(cal_data, ["age"], {"income": 100.0})
