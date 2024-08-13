from ..typing.MappedBaseModel import MappedBaseModel


class ClusterConfig(MappedBaseModel):
    """Config for clustering nodes

    Args:
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 10.
    """

    max_comm_size: int = 10
