from ..typing.MappedBaseModel import MappedBaseModel


class ClusterConfig(MappedBaseModel):
    """Config for clustering nodes

    Args:
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        levels (int, optional): Clusters can be split into clusters, how many levels should there
            be? Defaults to 2.
    """

    max_comm_size: int = 0
    levels: int = 2
