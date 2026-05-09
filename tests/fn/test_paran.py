"""Tests for moirais.fn.paran — Horn's parallel analysis."""

import pytest
import numpy as np
from moirais.fn import paran


class TestParan:
    """Tests for parallel analysis factor retention."""

    def test_returns_positive_int(self):
        """Should return an integer >= 1."""
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(100)
        data = np.column_stack([
            latent + rng.standard_normal(100) * 0.3
            for _ in range(5)
        ])
        result = paran(data, nsim=50, seed=42)
        assert isinstance(result, int)
        assert result >= 1

    def test_single_factor_data(self):
        """Strong single-factor data should suggest 1 factor."""
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(200)
        data = np.column_stack([
            latent + rng.standard_normal(200) * 0.2
            for _ in range(4)
        ])
        result = paran(data, nsim=50, seed=42)
        assert result == 1

    def test_two_factor_data(self):
        """Two-factor data should suggest >= 2 factors."""
        rng = np.random.default_rng(42)
        f1 = rng.standard_normal(300)
        f2 = rng.standard_normal(300)
        data = np.column_stack([
            f1 + rng.standard_normal(300) * 0.2,
            f1 + rng.standard_normal(300) * 0.2,
            f1 + rng.standard_normal(300) * 0.2,
            f2 + rng.standard_normal(300) * 0.2,
            f2 + rng.standard_normal(300) * 0.2,
            f2 + rng.standard_normal(300) * 0.2,
        ])
        result = paran(data, nsim=100, seed=42)
        assert result >= 2
