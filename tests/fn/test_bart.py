"""Tests for morie.fn.bart — Bartlett's test of sphericity."""

import pytest
import numpy as np
from morie.fn import bart
from morie.fn._containers import BrtRes


@pytest.fixture()
def correlated_data():
    """Correlated 4-item data — should reject sphericity H0."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(150)
    return np.column_stack([
        latent + rng.standard_normal(150) * 0.3
        for _ in range(4)
    ])


class TestBart:
    """Tests for Bartlett's sphericity test."""

    def test_correlated_rejects_h0(self, correlated_data):
        """Correlated data should reject H0 (p < 0.05)."""
        result = bart(correlated_data)
        assert result.pval < 0.05

    def test_returns_brtres(self, correlated_data):
        """Should return BrtRes dataclass."""
        result = bart(correlated_data)
        assert isinstance(result, BrtRes)

    def test_chisq_positive(self, correlated_data):
        """Chi-square statistic should be positive."""
        result = bart(correlated_data)
        assert result.chisq > 0

    def test_df_correct(self, correlated_data):
        """Degrees of freedom for k=4 items should be 4*3/2 = 6."""
        result = bart(correlated_data)
        assert result.df == 6

    def test_pval_range(self, correlated_data):
        """P-value should be between 0 and 1."""
        result = bart(correlated_data)
        assert 0 <= result.pval <= 1
