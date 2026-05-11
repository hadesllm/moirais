"""Tests for conditional_prob."""
import pytest
from morie.fn.condp import conditional_prob, condp


def test_basic():
    r = conditional_prob(0.1, 0.5)
    assert abs(r.estimate - 0.2) < 1e-10


def test_alias():
    assert condp is conditional_prob


def test_zero_marginal():
    with pytest.raises(ValueError):
        conditional_prob(0.1, 0.0)


def test_joint_exceeds_marginal():
    with pytest.raises(ValueError):
        conditional_prob(0.6, 0.5)
