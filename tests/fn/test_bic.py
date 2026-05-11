"""Test bic."""
import numpy as np
import pytest
from morie.fn.bic import bayesian_info_criterion


def test_bic_basic():
    r = bayesian_info_criterion(loglik=-100.0, n=50, k=3)
    assert r.name == "bic"
    assert r.value > 200


def test_bic_penalty():
    r1 = bayesian_info_criterion(loglik=-100.0, n=100, k=2)
    r2 = bayesian_info_criterion(loglik=-100.0, n=100, k=5)
    assert r2.value > r1.value
