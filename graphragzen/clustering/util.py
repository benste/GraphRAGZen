from collections import defaultdict
from typing import List

import networkx as nx
import pandas as pd


def _create_cluster_map(graph: nx.Graph) -> pd.DataFrame:
    """Creates a cluster -> nodes mapping

    Args:
        graph (nx.Graph): With has the feature 'cluster' added to the entities.

    Returns:
        pd.DataFrame
    """

    cluster_map = defaultdict(list)

    for node in graph.nodes(data=True):
        node_name = node[0]
        node_features = node[1]
        cluster = _string_represented_list_int_to_list(node_features.get("cluster", ""))
        # Make an entry for the main cluster and subclusters
        for i in range(1, len(cluster) + 1):
            # List is unhashable, so we represent the list as comma separated in a string
            cluster_name = _int_list_to_string_representation(cluster[:i])
            cluster_map[cluster_name].append(node_name)

    return pd.DataFrame({"cluster": cluster_map.keys(), "node_name": cluster_map.values()})


def _int_list_to_string_representation(input: List[int]) -> str:
    """Takes a list of integers and turns it into a comma separate string
    [1,2,3] -> "1,2,3"
    """
    return str(input).strip("[").strip("]").replace(" ", "")


def _string_represented_list_int_to_list(input: str) -> List[int]:
    """Takes a comma separate string representation of integers and turns it into list of integers
    "1,2,3" -> [1,2,3]
    """
    return [int(num) for num in input.split(",") if num]
