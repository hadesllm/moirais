# morie.fn — function file (hadesllm/morie)
"""Net migration rate."""

from ._containers import ESRes


def net_migration_rate(
    immigrants: int,
    emigrants: int,
    population: int,
    per: int = 1000,
) -> ESRes:
    """Compute net migration rate per *per* population.

    Parameters
    ----------
    immigrants : int
    emigrants : int
    population : int
    per : int

    Returns
    -------
    ESRes
    """
    if population <= 0:
        raise ValueError("population must be positive")

    net = immigrants - emigrants
    rate = net / population * per

    return ESRes(
        measure="net_migration_rate",
        estimate=float(rate),
        extra={"net_migration": net, "immigrants": immigrants, "emigrants": emigrants, "per": per},
    )


migrt = net_migration_rate


def cheatsheet() -> str:
    return "net_migration_rate({}) -> Net migration rate."
