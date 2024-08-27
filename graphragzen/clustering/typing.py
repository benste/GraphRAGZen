from graphragzen.prompts.default_prompts import cluster_description_prompts

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


class DescribeClustersConfig(MappedBaseModel):
    """Configuration for describing clusters

    Args:
        prompt (str, optional): The prompt to use for the LLM to describe a cluster
    """

    prompt: str = cluster_description_prompts.CLUSTER_DESCRIPTION_PROMPT
