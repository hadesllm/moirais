"""Test focal loss."""
import numpy as np
from moirais.fn.focal import focal


def test_focal_basic():
    """Test basic focal loss."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    loss = focal(y_true, y_pred)
    assert loss > 0


def test_focal_from_logits():
    """Test with logits."""
    y_true = np.array([0, 1])
    y_pred = np.array([0.0, 2.0])
    loss = focal(y_true, y_pred, from_logits=True)
    assert loss > 0


def test_focal_easy_vs_hard():
    """Test focal loss emphasizes hard examples."""
    y_true = np.array([1, 1])
    y_pred_easy = np.array([0.99, 0.5])
    loss_easy = focal(y_true, y_pred_easy, gamma=2.0)
    loss_hard = focal(np.array([1]), np.array([0.5]), gamma=2.0)
    assert loss_hard > loss_easy


def test_focal_alpha():
    """Test alpha parameter."""
    y_true = np.array([0, 1])
    y_pred = np.array([0.5, 0.5])
    loss1 = focal(y_true, y_pred, alpha=0.25)
    loss2 = focal(y_true, y_pred, alpha=0.75)
    assert loss1 != loss2
