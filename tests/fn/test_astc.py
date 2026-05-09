"""Tests for moirais.fn.astc — alert-state combination encoding."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.astc import astcmb as astc
from moirais.fn._containers import AstRes


@pytest.fixture()
def alert_df():
    """Synthetic alert data with Yes/No strings."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "unique_individual_id": [f"P{i:04d}" for i in range(n)],
        "end_fiscal_year": rng.choice([2020, 2021], n),
        "number_of_placements": rng.integers(1, 5, n),
        "mental_health_alert": rng.choice(["Yes", "No"], n),
        "suicide_risk_alert": rng.choice(["Yes", "No"], n),
        "suicide_watch_alert": rng.choice(["Yes", "No"], n),
    })


class TestAstcmb:
    """Tests for alert-state combination encoding."""

    def test_returns_astres(self, alert_df):
        """Should return AstRes dataclass."""
        result = astc(alert_df)
        assert isinstance(result, AstRes)

    def test_complexity_range(self, alert_df):
        """Complexity (ac) should be between 0 and 8."""
        result = astc(alert_df)
        assert (result.data["ac"] >= 0).all()
        assert (result.data["ac"] <= 8).all()

    def test_handles_yes_no_strings(self, alert_df):
        """Should correctly binarize Yes/No strings."""
        result = astc(alert_df)
        acols = [f"a{i}" for i in range(1, 9)]
        # At least some combo columns should have nonzero values
        assert result.data[acols].sum().sum() > 0

    def test_numeric_input(self):
        """Should also work with integer 0/1 columns."""
        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame({
            "unique_individual_id": [f"P{i:04d}" for i in range(n)],
            "end_fiscal_year": [2021] * n,
            "number_of_placements": [1] * n,
            "mental_health_alert": rng.integers(0, 2, n),
            "suicide_risk_alert": rng.integers(0, 2, n),
            "suicide_watch_alert": rng.integers(0, 2, n),
        })
        result = astc(df)
        assert isinstance(result, AstRes)
