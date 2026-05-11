# morie.fn — function file (hadesllm/morie)
"""Cluster sampling: select clusters, take all units within. 'We are what they grow beyond.'."""

from morie.sampling import cluster_sample as _fn

clsmp = _fn
cluster_sample = _fn


def cheatsheet() -> str:
    return "clsmp() -> Cluster sampling: select clusters, take all units within. 'W"
