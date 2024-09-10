from copy import deepcopy

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np

from .util import _create_cluster_map, _int_list_to_string_representation


def leiden(
    graph: nx.Graph,
    max_comm_size: int = 0,
    min_comm_size: int = 0,
    levels: int = 2,
) -> tuple:
    """Graph clustering using the Leiden algorithm (see: https://arxiv.org/abs/1810.08473)

    note: Clusters have levels, i.e. cluster 1 can be subdevided into multiple clusters.
        This is represented by a comma separated string, where each index is a level. e.g. cluster
        "2, 11" is the 11th subcluster of the 2nd cluster, while cluster "4, 11" is associated with
        main cluster "4" and has no relation with cluster "2, 11".

    Args:
        graph (nx.Graph):
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        min_comm_size (int, optional): Minimum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        levels (int, optional): Clusters can be split into clusters, how many levels should there
            be? Defaults to 2.

    Returns:
        tuple(nx.Graph, cluster_map): nx.Graph has the feature 'cluster' added to the entities.
            cluster_map maps for each cluster the nodes that belong to it.
    """
    clusters = _leiden(
        graph,
        max_comm_size=max_comm_size,
        min_comm_size=min_comm_size,
        levels=levels,
    )

    clustered_graph = deepcopy(graph)

    # Map back to graphnx
    for node_name, cluster in clusters.items():
        clustered_graph.nodes[node_name]["cluster"] = _int_list_to_string_representation(cluster)

    return clustered_graph, _create_cluster_map(clustered_graph)


def _leiden(
    graph: nx.Graph,
    max_comm_size: int = 0,
    min_comm_size: int = 0,
    levels: int = 2,
) -> dict:
    """Graph clustering using the Leiden algorithm (see: https://arxiv.org/abs/1810.08473)

    Args:
        graph (nx.Graph)
        max_comm_size (int, optional): Maximum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        min_comm_size (int, optional): Minimum number of nodes in one cluster. Defaults to 0 (no
            contraint).
        levels (int, optional): Clusters can be split into clusters, how many levels should there
            be? Defaults to 2.

    Returns:
        dict: {node_name: cluster, node_name2: cluster, etc.}
    """

    igraph = ig.Graph.from_networkx(graph)
    partitions = la.find_partition(
        igraph,
        la.ModularityVertexPartition,
        max_comm_size=max_comm_size,
        n_iterations=20,
    )

    nodes_list = np.asarray(igraph.vs["_nx_name"])
    clusters = dict()
    for partition, nodes in enumerate(partitions):
        # Nodes in igraph are indices. Let's map the cluster to the nx graph node name
        nodes_in_cluster = nodes_list[nodes].tolist()
        if len(nodes_in_cluster) >= min_comm_size:
            mapping = dict(zip(nodes_in_cluster, [[partition] for _ in range(len(nodes))]))
            clusters.update(mapping)

            if levels > 1:
                subgraph = graph.subgraph(nodes_in_cluster)
                subgrap_clusters = _leiden(
                    subgraph, max_comm_size=int(max_comm_size / 2), levels=levels - 1
                )
                for node, map in subgrap_clusters.items():
                    clusters[node] += map

    return clusters
