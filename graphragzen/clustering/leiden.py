from typing import Union

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np

from .typing import ClusterConfig


def _leiden(graph: nx.Graph, **kwargs: Union[dict, ClusterConfig]) -> dict:
    """Graph clustering using the Leiden algorithm (see: https://arxiv.org/abs/1810.08473)

    Args:
        graph (nx.Graph)
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        levels (int, optional): Clusters can be split into clusters, how many levels should there
            be? Defaults to 2.

    Returns:
        dict: {node_name: cluster, node_name2: cluster, etc.}
    """
    config = ClusterConfig(**kwargs)  # type: ignore

    igraph = ig.Graph.from_networkx(graph)
    partitions = la.find_partition(
        igraph,
        la.ModularityVertexPartition,
        max_comm_size=config.max_comm_size,
        n_iterations=10,
    )

    nodes_list = np.asarray(igraph.vs["_nx_name"])
    clusters = dict()
    for partition, nodes in enumerate(partitions):
        # Nodes in igraph are indices. Let's map the cluster to the nx graph node name
        nodes_in_custer = nodes_list[nodes].tolist()
        mapping = dict(zip(nodes_in_custer, [[partition]] * len(nodes)))
        clusters.update(mapping)

        if config.levels > 1:
            subgraph = graph.subgraph(nodes_in_custer)
            subgrap_clusters = _leiden(subgraph, kwargs={"levels": config.levels - 1})
            for node, map in subgrap_clusters.items():
                clusters[node] += map

    return clusters


def leiden(graph: nx.Graph, **kwargs: Union[dict, ClusterConfig]) -> nx.Graph:
    """Graph clustering using the Leiden algorithm (see: https://arxiv.org/abs/1810.08473)

    Args:
        graph (nx.Graph)
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        levels (int, optional): Clusters can be split into clusters, how many levels should there
            be? Defaults to 2.

    Returns:
        nx.Graph: With the feature 'cluster' added to the entities
    """
    config = ClusterConfig(**kwargs)  # type: ignore

    clusters = _leiden(graph, config=config)

    # Map back to graphnx
    for entity_name, cluster in clusters.items():
        graph.nodes[entity_name]["cluster"] = str(cluster)

    return graph
