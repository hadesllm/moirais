"""Tests for morie.fn.ps_wgt — Post-stratification weights."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.ps_wgt import poststratification_weights


@pytest.fixture()
def ps_data():
    """Sample with 3 strata and known population counts."""
    rng = np.random.default_rng(42)
    strata = rng.choice(["A", "B", "C"], size=200, p=[0.2, 0.5, 0.3])
    return pd.DataFrame({"stratum": strata, "y": rng.standard_normal(200)})


def test_returns_series(ps_data):
    """poststratification_weights returns a pd.Series."""
    pop = {"A": 1000, "B": 3000, "C": 2000}
    result = poststratification_weights(ps_data, "stratum", pop)
    assert isinstance(result, pd.Series)
    assert len(result) == len(ps_data)


def test_weights_positive(ps_data):
    """All post-stratification weights should be positive."""
    pop = {"A": 1000, "B": 3000, "C": 2000}
    result = poststratification_weights(ps_data, "stratum", pop)
    assert (result > 0).all()


def test_weighted_proportions_match_population(ps_data):
    """Weighted stratum proportions should match population proportions."""
    pop = {"A": 1000, "B": 3000, "C": 2000}
    N_total = sum(pop.values())
    w = poststratification_weights(ps_data, "stratum", pop)
    for stratum, N_h in pop.items():
        mask = ps_data["stratum"] == stratum
        weighted_frac = w[mask].sum() / w.sum()
        pop_frac = N_h / N_total
        assert weighted_frac == pytest.approx(pop_frac, abs=1e-10)


def test_missing_stratum_raises():
    """Strata in sample but not in population_counts should raise ValueError."""
    df = pd.DataFrame({"stratum": ["A", "B", "C"], "y": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing"):
        poststratification_weights(df, "stratum", {"A": 100, "B": 200})


def test_missing_column_raises():
    """Non-existent strata_col should raise ValueError."""
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="not found"):
        poststratification_weights(df, "stratum", {"A": 100})
