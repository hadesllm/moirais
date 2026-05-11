"""Tests for morie.fn.mcdo — McDonald's omega."""

import pytest
import numpy as np
from morie.fn import mcdo
from morie.fn._containers import OmgRes


@pytest.fixture()
def correlated_items():
    """5 items x 100 respondents from a single latent factor."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(100)
    return np.column_stack([
        latent + rng.standard_normal(100) * 0.3
        for _ in range(5)
    ])


class TestMcdo:
    """Tests for McDonald's omega total and hierarchical."""

    def test_omega_total_range(self, correlated_items):
        """Omega total should be between 0 and 1."""
        result = mcdo(correlated_items)
        assert isinstance(result, OmgRes)
        assert 0 <= result.total <= 1

    def test_omega_hier_bounded(self, correlated_items):
        """Omega hierarchical should be between 0 and 1."""
        result = mcdo(correlated_items)
        assert 0 <= result.hier <= 1

    def test_returns_omgres(self, correlated_items):
        """Should return OmgRes dataclass."""
        result = mcdo(correlated_items)
        assert isinstance(result, OmgRes)
        assert result.nf == 1

    def test_expvar_positive(self, correlated_items):
        """Explained variance should be positive for correlated items."""
        result = mcdo(correlated_items)
        assert result.expvar > 0
