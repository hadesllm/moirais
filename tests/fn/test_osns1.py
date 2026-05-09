"""Tests for moirais.fn.osns1 — Rosenbaum sensitivity bounds."""

from moirais.fn.osns1 import otis_sensitivity


class TestOtisSensitivity:
    def test_returns_dict(self, otis_df):
        result = otis_sensitivity(otis_df)
        assert isinstance(result, dict)

    def test_has_keys(self, otis_df):
        result = otis_sensitivity(otis_df)
        for k in ("gamma_values", "p_upper", "p_lower", "n_pairs"):
            assert k in result

    def test_gamma_1_pval_symmetric(self, otis_df):
        """At Gamma=1 (no bias), upper and lower should be equal."""
        result = otis_sensitivity(otis_df, gamma_range=[1.0])
        assert abs(result["p_upper"][0] - result["p_lower"][0]) < 0.01

    def test_monotonicity(self, otis_df):
        """p_upper should be non-decreasing with Gamma."""
        result = otis_sensitivity(otis_df)
        p_up = result["p_upper"]
        for i in range(1, len(p_up)):
            assert p_up[i] >= p_up[i - 1] - 0.001  # tolerance

    def test_n_pairs_positive(self, otis_df):
        result = otis_sensitivity(otis_df)
        assert result["n_pairs"] > 0
