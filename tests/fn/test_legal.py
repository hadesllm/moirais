"""Tests for moirais.fn.legal — legal BAC limit check."""

import pytest
from moirais.fn.legal import is_over_legal_limit as legal


class TestIsOverLegalLimit:
    """Tests for is_over_legal_limit."""

    def test_over_limit_returns_one(self):
        """BAC above 0.08 should return 1."""
        assert legal(0.10) == 1

    def test_under_limit_returns_zero(self):
        """BAC below 0.08 should return 0."""
        assert legal(0.05) == 0

    def test_at_limit_returns_one(self):
        """BAC exactly at 0.08 should return 1 (>= comparison)."""
        assert legal(0.08) == 1

    def test_zero_bac(self):
        """Zero BAC is under the limit."""
        assert legal(0.0) == 0

    def test_custom_limit(self):
        """Custom limit of 0.05 (e.g. Canada)."""
        assert legal(0.06, limit=0.05) == 1
        assert legal(0.04, limit=0.05) == 0
