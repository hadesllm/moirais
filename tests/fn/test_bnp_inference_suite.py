"""Integration tests for 30 Bayesian nonparametric inference functions."""

import numpy as np
import pytest


def test_all_bnp_functions_importable():
    """Test that all 30 BNP functions can be imported."""
    from morie.fn import (
        dpgen, dpprr, dpmix, pyprr, crpst, ibprc, stbrk, polya,
        bferg, bbstr, dpkde, gprgr, gpclf, gpkrn, gphyp, bspln,
        bwavl, lddst, bnpht, polyt, brnst, dpmdn, crppr, ewens,
        neale, slcmx, hdprc, bnpqs, postc, bcntr
    )

    # Verify they're callable
    functions = [
        dpgen, dpprr, dpmix, pyprr, crpst, ibprc, stbrk, polya,
        bferg, bbstr, dpkde, gprgr, gpclf, gpkrn, gphyp, bspln,
        bwavl, lddst, bnpht, polyt, brnst, dpmdn, crppr, ewens,
        neale, slcmx, hdprc, bnpqs, postc, bcntr
    ]

    assert len(functions) == 30
    for func in functions:
        assert callable(func)


def test_dpgen_returns_dict():
    """Test DP stick-breaking."""
    from morie.fn.dpgen import dirichlet_process_gen

    result = dirichlet_process_gen(alpha=1.0, n_clusters=10)

    assert isinstance(result, dict)
    assert "weights" in result
    assert np.all(result["weights"] > 0)


def test_crpst_returns_dict():
    """Test Chinese Restaurant Process."""
    from morie.fn.crpst import chinese_restaurant_process

    result = chinese_restaurant_process(n=50, alpha=1.0)

    assert isinstance(result, dict)
    assert "table_assignments" in result
    assert len(result["table_assignments"]) == 50


def test_bbstr_returns_dict():
    """Test Bayesian bootstrap."""
    from morie.fn.bbstr import bayesian_bootstrap

    x = np.random.normal(0, 1, 30)
    result = bayesian_bootstrap(x, n_boot=50)

    assert isinstance(result, dict)
    assert "posterior_mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result


def test_postc_basic():
    """Test posterior consistency rate."""
    from morie.fn.postc import posterior_consistency_rate

    rate = posterior_consistency_rate(n=100, dimension=1, smoothness=2.0)

    assert rate > 0
    assert rate < 1  # Should be a small rate


def test_bcntr_basic():
    """Test Bayesian contraction rate."""
    from morie.fn.bcntr import bayesian_contraction_rate

    result = bayesian_contraction_rate(n=100, dimension=1)

    assert isinstance(result, dict)
    assert "contraction_rate" in result
    assert result["contraction_rate"] > 0


def test_gprgr_basic():
    """Test Gaussian process regression."""
    from morie.fn.gprgr import gaussian_process_regression

    x = np.linspace(0, 10, 50).reshape(-1, 1)
    y = np.sin(x.ravel()) + np.random.normal(0, 0.1, 50)

    result = gaussian_process_regression(x, y)

    assert "mean" in result
    assert "var" in result
    assert len(result["mean"]) == 50


def test_dpmix_basic():
    """Test DP mixture model."""
    from morie.fn.dpmix import dirichlet_process_mixture

    x = np.concatenate([
        np.random.normal(-3, 1, 25),
        np.random.normal(3, 1, 25)
    ])

    result = dirichlet_process_mixture(x, alpha=1.0, n_iter=50)

    assert "cluster_assignments" in result
    assert "cluster_means" in result
    assert "n_clusters" in result


def test_polya_basic():
    """Test Pólya urn."""
    from morie.fn.polya import polya_urn

    result = polya_urn(
        n_draws=100,
        initial_colors={'red': 5, 'blue': 5},
        reinforcement=1
    )

    assert len(result["draws"]) == 100
    assert "proportions" in result


def test_ibprc_basic():
    """Test Indian buffet process."""
    from morie.fn.ibprc import indian_buffet_process

    result = indian_buffet_process(n=30, alpha=1.0)

    assert result["n_customers"] == 30
    assert result["features"].shape[0] == 30
    assert result["n_dishes"] > 0


def test_ewens_basic():
    """Test Ewens sampling."""
    from morie.fn.ewens import ewens_partition

    result = ewens_partition(n=50, theta=1.0)

    assert result["n"] == 50
    assert result["n_classes"] > 0


def test_neale_basic():
    """Test Neal's Algorithm 8."""
    from morie.fn.neale import neal_algorithm_8

    x = np.concatenate([
        np.random.normal(-2, 1, 20),
        np.random.normal(2, 1, 20)
    ])

    result = neal_algorithm_8(x, alpha=1.0, n_iter=20)

    assert "final_n_clusters" in result
    assert result["final_n_clusters"] > 0


def test_bnpqs_basic():
    """Test Bayesian nonparametric quantiles."""
    from morie.fn.bnpqs import bayesian_nonparametric_quantiles

    x = np.random.exponential(2, 100)
    result = bayesian_nonparametric_quantiles(
        x, quantiles=np.array([0.5]), n_boot=50
    )

    assert "quantile_estimates" in result
    assert 0.5 in {float(k) for k in result["quantile_estimates"].keys()}
