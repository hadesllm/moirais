"""Tests for moirais.fn.strat — Stratified mean estimator."""
import numpy as np
import pandas as pd

from moirais.fn.strat import stratified_mean, strat


def test_equal_strata():
    """Equal-sized strata: stratified mean = overall mean."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "y": rng.standard_normal(200),
        "stratum": np.repeat(["A", "B"], 100),
    })
    result = stratified_mean(df)
    overall = df["y"].mean()
    assert abs(result.value - overall) < 0.01


def test_weighted_mean():
    """With population sizes, the weighted mean should differ from simple mean."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "y": np.concatenate([rng.normal(10, 1, 50), rng.normal(20, 1, 50)]),
        "stratum": np.repeat(["A", "B"], 50),
    })
    # A is 90% of population
    result = stratified_mean(df, pop_sizes={"A": 900, "B": 100})
    assert result.value < 15  # Weighted toward A's mean of 10
    assert result.extra["n_strata"] == 2


def test_strat_alias():
    assert strat is stratified_mean
