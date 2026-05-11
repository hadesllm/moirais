"""Tests for morie.fn.kmo — Kaiser-Meyer-Olkin sampling adequacy."""

import pytest
import numpy as np
from morie.fn import kmo
from morie.fn._containers import KmoRes


@pytest.fixture()
def correlated_data():
    """Correlated 5-item data that should be factorable."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(200)
    return np.column_stack([
        latent + rng.standard_normal(200) * 0.3
        for _ in range(5)
    ])


class TestKmo:
    """Tests for Kaiser-Meyer-Olkin measure."""

    def test_correlated_data_high_msa(self, correlated_data):
        """Highly correlated data should have MSA > 0.5."""
        result = kmo(correlated_data)
        assert result.msa > 0.5

    def test_returns_kmores(self, correlated_data):
        """Should return KmoRes dataclass."""
        result = kmo(correlated_data)
        assert isinstance(result, KmoRes)

    def test_items_dict(self, correlated_data):
        """Items dict should have one entry per variable."""
        result = kmo(correlated_data)
        assert isinstance(result.items, dict)
        assert len(result.items) == 5

    def test_item_msa_range(self, correlated_data):
        """Per-item MSA should be between 0 and 1."""
        result = kmo(correlated_data)
        for v in result.items.values():
            assert 0 <= v <= 1
