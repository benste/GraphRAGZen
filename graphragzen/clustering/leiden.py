from typing import Any

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np

from .typing import ClusterConfig


def leiden(graph: nx.Graph, **kwargs: Any) -> nx.Graph:
    """Graph clustering using the Leiden algorithm (see: https://arxiv.org/abs/1810.08473)

    Args:
        graph (nx.Graph)
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 10.

    Returns:
        nx.Graph: With the feature 'cluster' added to the entities
    """
    config = ClusterConfig(**kwargs)  # type: ignore

    igraph = ig.Graph.from_networkx(graph)
    partition = la.find_partition(
        igraph, la.ModularityVertexPartition, max_comm_size=config.max_comm_size
    )

    nodes_list = np.asarray(igraph.vs["_nx_name"])
    partitions = dict()
    for partition, nodes in enumerate(
        la.find_partition(igraph, partition_type=la.ModularityVertexPartition, seed=0)
    ):
        mapping = dict(zip(nodes_list[nodes], [partition] * len(nodes)))
        partitions.update(mapping)

    for entity_name, cluster in partitions.items():
        graph.nodes[entity_name]["cluster"] = str(cluster)

    return graph
