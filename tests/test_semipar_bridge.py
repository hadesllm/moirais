"""Tests for morie.semipar_bridge — both C and NumPy backends.

Tests verify mathematical correctness of:
- Nadaraya-Watson kernel regression
- Local linear regression
- Kernel density estimation (all 5 kernel types)
- Silverman bandwidth selection
- LOOCV bandwidth selection
- Conditional moments
- SemiparKernels class interface
"""

from __future__ import annotations

import numpy as np
import pytest

from morie.semipar_bridge import (
    KERNEL_BIWEIGHT,
    KERNEL_EPANECHNIKOV,
    KERNEL_GAUSSIAN,
    KERNEL_TRIANGULAR,
    KERNEL_UNIFORM,
    SemiparKernels,
    is_available,
    kde,
    kernel_cond_moments,
    kernel_eval,
    local_linear,
    loocv_bandwidth,
    nw_regression,
    silverman_bandwidth,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sine_data(rng):
    n = 200
    x = np.sort(rng.uniform(0, 2 * np.pi, n))
    y = np.sin(x) + 0.2 * rng.standard_normal(n)
    x_eval = np.linspace(0.5, 2 * np.pi - 0.5, 50)
    return x, y, x_eval


@pytest.fixture
def normal_data(rng):
    return rng.standard_normal(500)


class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(is_available(), bool)


class TestKernelEval:
    def test_gaussian_at_zero(self):
        k = kernel_eval(0.0, KERNEL_GAUSSIAN)
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        assert abs(k - expected) < 1e-10

    def test_gaussian_symmetry(self):
        assert abs(kernel_eval(1.5) - kernel_eval(-1.5)) < 1e-10

    def test_epanechnikov_at_zero(self):
        assert abs(kernel_eval(0.0, KERNEL_EPANECHNIKOV) - 0.75) < 1e-10

    def test_epanechnikov_outside(self):
        assert kernel_eval(1.5, KERNEL_EPANECHNIKOV) == 0.0

    def test_uniform_inside(self):
        assert abs(kernel_eval(0.5, KERNEL_UNIFORM) - 0.5) < 1e-10

    def test_uniform_outside(self):
        assert kernel_eval(1.5, KERNEL_UNIFORM) == 0.0

    def test_triangular_at_zero(self):
        assert abs(kernel_eval(0.0, KERNEL_TRIANGULAR) - 1.0) < 1e-10

    def test_triangular_outside(self):
        assert kernel_eval(2.0, KERNEL_TRIANGULAR) == 0.0

    def test_biweight_at_zero(self):
        assert abs(kernel_eval(0.0, KERNEL_BIWEIGHT) - 15.0 / 16.0) < 1e-10

    def test_biweight_outside(self):
        assert kernel_eval(1.1, KERNEL_BIWEIGHT) == 0.0


class TestNWRegression:
    def test_shape(self, sine_data):
        x, y, x_eval = sine_data
        bw = silverman_bandwidth(x)
        y_hat = nw_regression(x, y, x_eval, bw)
        assert y_hat.shape == x_eval.shape

    def test_recovers_sine(self, sine_data):
        x, y, x_eval = sine_data
        bw = 0.5
        y_hat = nw_regression(x, y, x_eval, bw)
        true_vals = np.sin(x_eval)
        rmse = np.sqrt(np.mean((y_hat - true_vals) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} too high for sine recovery"

    def test_constant_function(self):
        x = np.linspace(0, 1, 100)
        y = np.full(100, 5.0)
        x_eval = np.array([0.25, 0.5, 0.75])
        y_hat = nw_regression(x, y, x_eval, 0.2)
        np.testing.assert_allclose(y_hat, 5.0, atol=1e-6)

    def test_single_eval_point(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        x_eval = np.array([2.0])
        y_hat = nw_regression(x, y, x_eval, 1.0)
        assert y_hat.shape == (1,)
        assert abs(y_hat[0] - 2.0) < 0.5

    def test_dtype_float64(self, sine_data):
        x, y, x_eval = sine_data
        y_hat = nw_regression(x, y, x_eval, 0.5)
        assert y_hat.dtype == np.float64


class TestLocalLinear:
    def test_shape_no_slope(self, sine_data):
        x, y, x_eval = sine_data
        y_hat = local_linear(x, y, x_eval, 0.5, return_slope=False)
        assert y_hat.shape == x_eval.shape

    def test_shape_with_slope(self, sine_data):
        x, y, x_eval = sine_data
        result = local_linear(x, y, x_eval, 0.5, return_slope=True)
        assert isinstance(result, tuple)
        y_hat, beta_hat = result
        assert y_hat.shape == x_eval.shape
        assert beta_hat.shape == x_eval.shape

    def test_recovers_sine(self, sine_data):
        x, y, x_eval = sine_data
        y_hat = local_linear(x, y, x_eval, 0.5, return_slope=False)
        true_vals = np.sin(x_eval)
        rmse = np.sqrt(np.mean((y_hat - true_vals) ** 2))
        assert rmse < 0.3, f"RMSE {rmse:.4f} too high"

    def test_slope_on_linear(self):
        x = np.linspace(0, 10, 200)
        y = 2.0 * x + 1.0 + 0.01 * np.random.default_rng(99).standard_normal(200)
        x_eval = np.array([3.0, 5.0, 7.0])
        y_hat, beta_hat = local_linear(x, y, x_eval, 1.0, return_slope=True)
        np.testing.assert_allclose(beta_hat, 2.0, atol=0.3)

    def test_less_boundary_bias_than_nw(self):
        x = np.linspace(0, 1, 100)
        y = x.copy()
        x_eval = np.array([0.02, 0.98])
        bw = 0.15
        y_hat_ll = local_linear(x, y, x_eval, bw, return_slope=False)
        y_hat_nw = nw_regression(x, y, x_eval, bw)
        err_ll = np.abs(y_hat_ll - x_eval)
        err_nw = np.abs(y_hat_nw - x_eval)
        assert err_ll.mean() <= err_nw.mean() + 0.01


class TestKDE:
    def test_shape(self, normal_data):
        x_eval = np.linspace(-3, 3, 50)
        bw = silverman_bandwidth(normal_data)
        density = kde(normal_data, x_eval, bw)
        assert density.shape == x_eval.shape

    def test_nonnegative(self, normal_data):
        x_eval = np.linspace(-4, 4, 100)
        density = kde(normal_data, x_eval, 0.5)
        assert np.all(density >= 0)

    def test_integrates_to_one(self, normal_data):
        x_eval = np.linspace(-6, 6, 500)
        bw = silverman_bandwidth(normal_data)
        density = kde(normal_data, x_eval, bw)
        dx = x_eval[1] - x_eval[0]
        integral = np.trapezoid(density, dx=dx)
        assert abs(integral - 1.0) < 0.1, f"Integral = {integral:.4f}"

    def test_peak_near_zero_for_standard_normal(self, normal_data):
        x_eval = np.linspace(-4, 4, 200)
        bw = silverman_bandwidth(normal_data)
        density = kde(normal_data, x_eval, bw)
        peak_idx = np.argmax(density)
        peak_x = x_eval[peak_idx]
        assert abs(peak_x) < 0.5, f"Peak at {peak_x}"

    @pytest.mark.parametrize("kernel_type", [
        KERNEL_GAUSSIAN,
        KERNEL_EPANECHNIKOV,
        KERNEL_UNIFORM,
        KERNEL_TRIANGULAR,
        KERNEL_BIWEIGHT,
    ])
    def test_all_kernels_nonnegative(self, normal_data, kernel_type):
        x_eval = np.linspace(-4, 4, 50)
        density = kde(normal_data, x_eval, 0.5, kernel_type=kernel_type)
        assert np.all(density >= 0)


class TestSilvermanBandwidth:
    def test_positive(self, normal_data):
        bw = silverman_bandwidth(normal_data)
        assert bw > 0

    def test_scales_with_spread(self):
        narrow = np.random.default_rng(1).standard_normal(100) * 0.1
        wide = np.random.default_rng(1).standard_normal(100) * 10.0
        bw_narrow = silverman_bandwidth(narrow)
        bw_wide = silverman_bandwidth(wide)
        assert bw_wide > bw_narrow

    def test_scales_with_n(self):
        rng = np.random.default_rng(7)
        small = rng.standard_normal(50)
        large = rng.standard_normal(5000)
        bw_small = silverman_bandwidth(small)
        bw_large = silverman_bandwidth(large)
        assert bw_large < bw_small

    def test_constant_data(self):
        x = np.ones(100)
        bw = silverman_bandwidth(x)
        assert bw > 0


class TestLOOCV:
    def test_returns_positive(self, sine_data):
        x, y, _ = sine_data
        bw = loocv_bandwidth(x, y, n_grid=10)
        assert bw > 0

    def test_within_range(self, sine_data):
        x, y, _ = sine_data
        h_rot = silverman_bandwidth(x)
        bw_min = 0.1 * h_rot
        bw_max = 2.0 * h_rot
        bw = loocv_bandwidth(x, y, bw_min=bw_min, bw_max=bw_max, n_grid=10)
        assert bw_min <= bw <= bw_max

    def test_explicit_range(self):
        rng = np.random.default_rng(77)
        x = np.sort(rng.uniform(0, 5, 50))
        y = np.sin(x) + 0.1 * rng.standard_normal(50)
        bw = loocv_bandwidth(x, y, bw_min=0.1, bw_max=2.0, n_grid=10)
        assert 0.1 <= bw <= 2.0


class TestCondMoments:
    def test_mean_shape(self, sine_data):
        x, y, x_eval = sine_data
        mean_out = kernel_cond_moments(x, y, x_eval, 0.5, return_variance=False)
        assert mean_out.shape == x_eval.shape

    def test_mean_and_var_shape(self, sine_data):
        x, y, x_eval = sine_data
        result = kernel_cond_moments(x, y, x_eval, 0.5, return_variance=True)
        assert isinstance(result, tuple)
        mean_out, var_out = result
        assert mean_out.shape == x_eval.shape
        assert var_out.shape == x_eval.shape

    def test_variance_nonnegative(self, sine_data):
        x, y, x_eval = sine_data
        _, var_out = kernel_cond_moments(x, y, x_eval, 0.5, return_variance=True)
        assert np.all(var_out >= 0)

    def test_constant_has_zero_variance(self):
        x = np.linspace(0, 1, 100)
        y = np.full(100, 3.14)
        x_eval = np.array([0.25, 0.5, 0.75])
        mean_out, var_out = kernel_cond_moments(x, y, x_eval, 0.2, return_variance=True)
        np.testing.assert_allclose(mean_out, 3.14, atol=1e-6)
        np.testing.assert_allclose(var_out, 0.0, atol=1e-6)

    def test_mean_matches_nw(self, sine_data):
        x, y, x_eval = sine_data
        bw = 0.5
        mean_out = kernel_cond_moments(x, y, x_eval, bw, return_variance=False)
        y_hat = nw_regression(x, y, x_eval, bw)
        np.testing.assert_allclose(mean_out, y_hat, atol=1e-10)


class TestSemiparKernelsClass:
    def test_backend_string(self):
        sk = SemiparKernels()
        assert sk.backend in ("c", "numpy")

    def test_available_matches_module(self):
        sk = SemiparKernels()
        assert sk.available == is_available()

    def test_nw_regression(self, sine_data):
        x, y, x_eval = sine_data
        sk = SemiparKernels()
        y_hat = sk.nw_regression(x, y, x_eval, 0.5)
        assert y_hat.shape == x_eval.shape
        y_hat_mod = nw_regression(x, y, x_eval, 0.5)
        np.testing.assert_allclose(y_hat, y_hat_mod)

    def test_local_linear(self, sine_data):
        x, y, x_eval = sine_data
        sk = SemiparKernels()
        y_hat, beta_hat = sk.local_linear(x, y, x_eval, 0.5)
        assert y_hat.shape == x_eval.shape
        assert beta_hat.shape == x_eval.shape

    def test_kde_gaussian(self, normal_data):
        sk = SemiparKernels()
        x_eval = np.linspace(-3, 3, 30)
        density = sk.kde(normal_data, x_eval, 0.5, kernel="gaussian")
        assert density.shape == x_eval.shape
        assert np.all(density >= 0)

    def test_kde_epanechnikov(self, normal_data):
        sk = SemiparKernels()
        x_eval = np.linspace(-3, 3, 30)
        density = sk.kde(normal_data, x_eval, 0.5, kernel="epanechnikov")
        assert np.all(density >= 0)

    def test_kde_invalid_kernel(self, normal_data):
        sk = SemiparKernels()
        x_eval = np.linspace(-3, 3, 10)
        with pytest.raises(ValueError, match="Unknown kernel"):
            sk.kde(normal_data, x_eval, 0.5, kernel="invalid")

    def test_silverman(self, normal_data):
        sk = SemiparKernels()
        bw = sk.silverman_bandwidth(normal_data)
        assert bw > 0
        assert abs(bw - silverman_bandwidth(normal_data)) < 1e-15

    def test_loocv(self, sine_data):
        x, y, _ = sine_data
        sk = SemiparKernels()
        bw = sk.loocv_bandwidth(x, y, n_grid=5)
        assert bw > 0


class TestEdgeCases:
    def test_two_points_nw(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        x_eval = np.array([0.5])
        y_hat = nw_regression(x, y, x_eval, 0.5)
        assert y_hat.shape == (1,)
        assert 0.0 <= y_hat[0] <= 1.0

    def test_identical_x_values(self):
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        x_eval = np.array([1.0])
        y_hat = nw_regression(x, y, x_eval, 0.5)
        assert abs(y_hat[0] - 3.5) < 1e-6

    def test_silverman_two_points(self):
        bw = silverman_bandwidth(np.array([0.0, 1.0]))
        assert bw > 0

    def test_integer_input(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        x_eval = np.array([3])
        y_hat = nw_regression(x, y, x_eval, 1.0)
        assert y_hat.dtype == np.float64
