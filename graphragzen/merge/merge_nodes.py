import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from scipy.spatial.distance import pdist, squareform


def contain_date(input: str) -> bool:
    """Check if string contains a date

    Args:
        input (str):

    Returns:
        bool: True if string contains a date
    """
    date_regexes = [
        r"\d{1,2}(\/|\.|-)\d{1,2}(\/|\.|-)\d{2,4}",
        r"\d{2,4}(\/|\.|-)\d{1,2}(\/|\.|-)\d{1,2}",
    ]

    if any(re.search(regex, input) for regex in date_regexes):
        return True

    return False


def isempty(input: str) -> bool:
    """Check if string is empty or None

    Args:
        input (str):

    Returns:
        bool:  True if string is empty or None
    """
    if not input:
        return True

    return False


def find_similar_nodes(
    nodes: list[str],
    embedding_vectors: np.ndarray,
    min_similarity: float = 0.975,
    embedding_source: Optional[list] = None,
) -> pd.DataFrame:
    """Finds nodes who's embedding vectors are > min_similarity.

    Args:
        nodes (list[str]): List of node names.
        embedding_vectors (np.ndarray): The embedding vectors of the features. Should have shape
            (num_nodes, embedding_vector_size).
        min_similarity (float, optional): Minimum similarity for 2 nodes to be concidered similar.
            Defaults to 0.975.
        embedding_source (list, optional): The raw text that was used to create the embedding
            vectors. Will be added to the returned report. Defaults to None.

    Returns:
        pd.DataFrame: Contains columns
            - 'nodes': node pair that exceeds min_similarity. tuple; (node1, node2)
            - 'similarity_score': How similar the feature of the two nodes are
            - 'features': Raw text of the features of each node. Only populated if
                'embedding_source' was given as an input.
    """

    if embedding_source is None:
        embedding_source = [None] * embedding_vectors.shape[0]

    similarity_score = 1 - squareform(pdist(embedding_vectors, metric="cosine"))
    np.fill_diagonal(similarity_score, 0)

    similar_index = list(zip(*np.where(similarity_score > min_similarity)))
    similar_index = [tuple(set(pair)) for pair in similar_index]

    merge_nodes_map = defaultdict(list)
    for pair in similar_index:
        merge_nodes_map["nodes"].append([nodes[pair[0]], nodes[pair[1]]])
        merge_nodes_map["similarity_score"].append(similarity_score[pair[0], pair[1]])
        merge_nodes_map["features"].append(
            {  # type: ignore
                nodes[pair[0]]: embedding_source[pair[0]],
                nodes[pair[1]]: embedding_source[pair[1]],  # type: ignore
            }
        )

    merge_report = pd.DataFrame(merge_nodes_map)

    # Drop duplicate similar nodes detected
    merge_report["nodes"] = merge_report["nodes"].apply(sorted)
    merge_report.drop_duplicates(subset="nodes", inplace=True)

    return merge_report


def _merge_nodes(
    graph: nx.Graph, nodes_to_merge: List[Tuple[str, str]], feature_delimiter: str = "\n"
) -> nx.Graph:
    """Merge nodes in a graph; transfers edges from node2 to node1 and appending the features
    of node1 with the features of node2

    Args:
        graph (nx.Graph): Graph of whom some nodes need merging
        nodes_to_merge (List[Tuple[str]]): Tuples of nodes that need merging.
        feature_delimiter (str, optional): Features are concatenated using this delimiter.
            Defaults to '\\n'.

    Returns:
        nx.Graph: New graph with merged nodes
    """
    merged_graph = deepcopy(graph)
    nodes_to_merge = deepcopy(nodes_to_merge)

    merge_map: dict = {}  # keeps track of which node has been merged into which other node
    for nodes in nodes_to_merge:
        # A node might have already been merged with another node in the loop, we'll assign the node
        # it was merged into
        while nodes[0] in merge_map:
            nodes[0] = merge_map.get(nodes[0], nodes[0])  # type: ignore

        while nodes[1] in merge_map:
            nodes[1] = merge_map.get(nodes[1], nodes[1])  # type: ignore

        # Theoretically we can now end-up with the same node wanting to merge with itself, let's
        # not do that
        if nodes[0] != nodes[1]:
            # Get the node info from the graph
            base_node = merged_graph.nodes[nodes[0]]
            similar_node = merged_graph.nodes[nodes[1]]

            # Merge attributes of similar node into base node
            for feature in base_node.keys():
                if feature != "type":
                    base_node[feature] += feature_delimiter + similar_node[feature]

            # Add the edges of similar node to base node
            base_node_edges = list(merged_graph.edges([nodes[0]]))
            similar_node_edges = list(merged_graph.edges([nodes[1]]))
            for new_edge in similar_node_edges:
                if new_edge not in base_node_edges:
                    source = nodes[0]
                    target = new_edge[1]
                    edge_data = merged_graph.edges[new_edge]
                    merged_graph.add_edge(
                        source,
                        target,
                        weight=edge_data.get("weight", 0),
                        description=edge_data.get("description,", ""),
                        source_id=edge_data.get("source_id", ""),
                    )

            # Remove one of the nodes
            merge_map[nodes[1]] = nodes[0]
            merged_graph.remove_node(nodes[1])

    return merged_graph


def merge_similar_graph_nodes(
    graph: nx.Graph,
    embedding_model: BaseEmbedder,
    merge_report: Optional[pd.DataFrame] = None,
    extra_features_to_compare: list = ["description"],
    min_similarity: float = 0.975,
    filter_functions: list = [contain_date, isempty],
    feature_delimiter: str = "\n",
    dry_run: bool = False,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Merge nodes in a graph that are very similar to each other. Similarity is judged using text
    embeddings of the node names and selected features.

    Args:
        graph (nx.Graph): The graph to check for similar nodes.
        embedding_model (BaseEmbedder): The model that embeds text features of the nodes.
        merge_report (pd.DataFrame, optional): If dry_run is set to True this function
            returns a report stating which nodes would me merged. When supplying this report it will
            be used to merge nodes, no new similar nodes will be searched. This is usefull if you
            want to check what nodes will be merges beforehand and make adjustments if necessary.
            Defaults to None.
        extra_features_to_compare (list, optional): Other than the name of the nodes, which features
            should be text embedded and compared for similarity. Defaults to ["description"].
        min_similarity (float, optional): The minimum similarity to consider two nodes similar.
            Defaults to 0.975.
        filter_functions (list, optional): These function determine per node, per feature, if it
            should be concidered for comparing to other nodes for similarity.
            When any of these functions, supplied with the node feature, returns True, the feature
            for this node will not be used to find similar nodes.
            Defaults to [contain_date, isempty].
        feature_delimiter (str, optional): When nodes are merged features are concatenated using
            this delimiter. Defaults to '\\n'.
        dry_run (bool, optional): If True, will return the original, unmodified graph, with a report
            stating which nodes would me merged if dry_run was False. Defaults to False.

    Returns:
        Tuple[nx.graph, pd.DataFrame]: (graph, report stating which nodes were merged)
    """

    if merge_report is None:
        # Create embeddings
        features = ["node_name"] + extra_features_to_compare
        merge_reports = []
        for feature in features:
            embeddings_map = defaultdict(list)
            for node in graph.nodes(data=True):
                if feature == "node_name":
                    to_embed = node[0]
                else:
                    to_embed = node[1][feature]

                # Check if this passes the filters
                if not any(func(to_embed) or func(node[0]) for func in filter_functions):
                    embeddings_map["embedding_source"].append(to_embed)
                    embeddings_map["nodes"].append(node[0])

            # Add nodes who's name are exactly the same if lowered
            if feature == "node_name":
                name_counts = Counter([node_name.lower() for node_name in graph.nodes])
                same_name = [name for name, count in name_counts.items() if count > 1]
                for node in graph.nodes:
                    if node.lower() in same_name and node not in embeddings_map["nodes"]:
                        embeddings_map["embedding_source"].append(node.lower())
                        embeddings_map["nodes"].append(node)

            # Embed the features
            print(f"text embedding node {feature}s to find semantically similar nodes to merge")
            embeddings_map["embedding_vectors"] = embedding_model.embed(  # type: ignore
                embeddings_map["embedding_source"], task="embed_document", show_progress_bar=True
            )

            # Find the nodes that are > min_similarity similar
            merge_reports.append(
                find_similar_nodes(
                    min_similarity=min_similarity,
                    **embeddings_map,  # type: ignore
                )
            )
            merge_reports[-1]["compared_feature"] = feature

        merge_report = pd.concat(merge_reports)
        merge_report["nodes"] = merge_report["nodes"].apply(sorted)
        merge_report.drop_duplicates(subset="nodes", inplace=True)

    if dry_run:
        # Return the merge report now with the unchanged graph
        return graph, merge_report

    # Merge nodes
    graph = _merge_nodes(graph, merge_report.nodes.tolist(), feature_delimiter=feature_delimiter)

    return graph, merge_report
