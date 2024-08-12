import leidenalg as la
import igraph as ig
import networkx as nx
import numpy as np

from graphragzen.typing import ClusterConfig

def leiden(graph: nx.Graph, config: ClusterConfig) -> nx.Graph:
    igraph = ig.Graph.from_networkx(graph)
    partition = la.find_partition(igraph, la.ModularityVertexPartition, max_comm_size=config.max_comm_size)
    
    nodes_list = np.asarray(igraph.vs["_nx_name"])
    partitions = dict()
    for partition, nodes in enumerate(la.find_partition(igraph, partition_type=la.ModularityVertexPartition, seed=0)):
        mapping = dict(zip(nodes_list[nodes], [partition]*len(nodes)))
        partitions.update(mapping)
        
    for entity_name, cluster in partitions.items():
        graph.nodes[entity_name]['cluster'] = str(cluster)
        
    return graph
    