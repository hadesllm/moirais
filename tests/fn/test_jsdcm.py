"""Test joint_sparse_decompose (jsdcm)."""
import numpy as np
from moirais.fn.jsdcm import joint_sparse_decompose, jsdcm
from moirais.fn._containers import DescriptiveResult


class TestJsdcm:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(20)
        x2 = rng.standard_normal(20)
        result = joint_sparse_decompose([x1, x2], lambda_=0.5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "joint_sparse_decompose"

    def test_joint_support(self):
        x1 = np.zeros(10)
        x2 = np.zeros(10)
        x1[0] = 5.0
        x2[0] = 3.0
        x1[5] = -2.0
        x2[5] = -4.0
        r = joint_sparse_decompose([x1, x2], lambda_=0.5)
        assert r.value <= 10

    def test_high_lambda_zeros(self):
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(8) * 0.01
        x2 = rng.standard_normal(8) * 0.01
        r = joint_sparse_decompose([x1, x2], lambda_=10.0)
        assert r.value == 0

    def test_coefficients_shape(self):
        rng = np.random.default_rng(42)
        signals = [rng.standard_normal(15) for _ in range(3)]
        r = joint_sparse_decompose(signals, lambda_=0.1)
        assert r.extra["coefficients"].shape == (15, 3)

    def test_alias(self):
        assert jsdcm is joint_sparse_decompose
