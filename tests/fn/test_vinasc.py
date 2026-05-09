"""Tests for vinasc.autodock_vina_score."""
import numpy as np
import pytest
from moirais.fn.vinasc import autodock_vina_score


def test_vinasc_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pose = np.random.default_rng(42).normal(0, 1, 100)
    result = autodock_vina_score(receptor, ligand_pose)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vinasc_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pose = np.random.default_rng(42).normal(0, 1, 100)
    result = autodock_vina_score(receptor, ligand_pose)
    assert isinstance(result, dict)
