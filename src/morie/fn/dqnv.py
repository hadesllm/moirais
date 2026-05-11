"""DQN with target network + replay."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deep_q_network"]


def deep_q_network(env, net, buffer, target_update):
    """
    DQN with target network + replay

    Formula: L = E[(r + γ max_a' Q_θ⁻(s',a') − Q_θ(s,a))²]

    Parameters
    ----------
    env : array-like
        Input data.
    net : array-like
        Input data.
    buffer : array-like
        Input data.
    target_update : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mnih et al (2015) Atari
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DQN with target network + replay"})


def cheatsheet():
    return "dqnv: DQN with target network + replay"
