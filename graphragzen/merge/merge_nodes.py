import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple
from uuid import uuid4

import networkx as nx
import numpy as np
import pandas as pd
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SearchRequest,
    VectorParams,
)


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
    feature_names: list[str],
    embedding_vectors: np.ndarray,
    min_similarity: float = 0.95,
    embedding_source: Optional[list] = None,
) -> pd.DataFrame:
    """Finds nodes who's embedding vectors are > min_similarity. Does this for each unique feature
    in feature_names.

    Args:
        nodes (list[str]): List of node names.
        feature_names (list[str]): List of the features of the nodes that were imbedded
        embedding_vectors (np.ndarray): The embedding vectors of the features. Should have shape
            (num_nodes, embedding_vector_size).
        min_similarity (float, optional): Minimum similarity for 2 nodes to be concidered similar.
            Defaults to 0.95.
        embedding_source (list, optional): The raw text that was used to create the embedding
            vectors. Will be added to the returned report. Defaults to None.

    Returns:
        pd.DataFrame: Contains columns
            - 'nodes': nodes that are similar, tuple, (node1, node2)
            - 'similarity_score': How similar the feature of the the two nodes is
            - 'compared_feature': Which feature of the nodes was compared
            - 'features': Raw text of the feature of each node. Only populated if 'embedding_source'
                was given as an input.
    """

    # Use a quick in-memory qdrant client, don't need to save the vectors to DB for this purpose
    vector_db_client = QdrantClient(":memory:")
    collection_name = "node_merging"
    vector_db_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_vectors.shape[1], distance=Distance.COSINE),
    )

    # Add embeddings to vector DB
    if not embedding_source:
        embedding_source = [None] * len(nodes)

    payloads = [
        {"node": node, "feature": feature, "source": source}
        for node, feature, source in zip(nodes, feature_names, embedding_source)
    ]

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload=payload,
        )
        for vector, payload in zip(embedding_vectors, payloads)
    ]

    vector_db_client.upsert(collection_name=collection_name, points=points)

    # For each feature, find the nodes that are > min_similarity similar
    merge_nodes_map = defaultdict(list)
    for feature in set(feature_names):
        print(f"finding pairs of nodes with similar '{feature}'")
        query_filter = Filter(must=[FieldCondition(key="feature", match=MatchValue(value=feature))])

        # Only compare vectors that adhere to the feature
        filtered_points = vector_db_client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=10**20,  # basically unlimited
            with_vectors=True,
        )
        filtered_payload, filtered_vectors = zip(
            *[(record.payload, record.vector) for record in filtered_points[0]]
        )

        requests = [
            SearchRequest(
                vector=vector,
                filter=query_filter,
                limit=5,
                score_threshold=min_similarity,
                with_payload=True,
            )
            for vector in filtered_vectors
        ]

        results = vector_db_client.search_batch(collection_name=collection_name, requests=requests)

        # Create a report on which nodes are similar, the compared feature, and the similarity score
        for result, base_payload in zip(results, filtered_payload):
            base_node = base_payload["node"]
            for match in result:
                similar_node = match.payload["node"]  # type: ignore
                if base_node != similar_node:
                    merge_nodes_map["nodes"].append((base_node, similar_node))
                    merge_nodes_map["similarity_score"].append(match.score)  # type: ignore
                    merge_nodes_map["compared_feature"].append(feature)  # type: ignore
                    merge_nodes_map["features"].append(
                        {  # type: ignore
                            base_node: base_payload["source"],
                            similar_node: match.payload["source"],  # type: ignore
                        }
                    )

    # Drop duplicate similar nodes detected
    merge_report = pd.DataFrame(merge_nodes_map)
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
                    base_node[feature] += "\n" + similar_node[feature]

            # Add the edges of similar node to base node
            base_node_edges = merged_graph.edges([nodes[0]])
            similar_node_edges = merged_graph.edges([nodes[1]])
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
    min_similarity: float = 0.95,
    filter_functions: list = [contain_date, isempty],
    feature_delimiter: str = "\n",
    dry_run: bool = False,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Merge nodes in a graph that are very similar to each other, using text embeddings of the node
    names and selected features.

    Args:
        graph (nx.Graph): The graph to check for similar nodes.
        embedding_model (BaseEmbedder): The model to embed text features of the nodes.
        merge_report (Optional[pd.DataFrame], optional): If dry_run is set to True this function
            returns a report stating which nodes would me merged. When supplying this report it will
            be used to merge nodes, no new similar nodes will be searched. This is usefull if you
            want to check what nodes will be merges beforehand and make adjustments if necessary.
            Defaults to None.
        extra_features_to_compare (list, optional): Other than the name of the nodes, which features
            should be text embedded and compared for similarity. Defaults to ["description"].
        min_similarity (float, optional): The minimum similarity to consider two nodes similar.
            Defaults to 0.95.
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
        embeddings_map = defaultdict(list)
        features = ["node_name"] + extra_features_to_compare
        for feature in features:
            for node in graph.nodes(data=True):
                if feature == "node_name":
                    to_embed = node[0]
                else:
                    to_embed = node[1][feature]

                # Check if this passes the filters
                if not any(func(to_embed) for func in filter_functions):
                    embeddings_map["embedding_source"].append(to_embed)
                    embeddings_map["nodes"].append(node[0])
                    embeddings_map["feature_names"].append(feature)

        print("text embedding nodes for finding nodes to merge")
        embeddings_map["embedding_vectors"] = embedding_model.embed(  # type: ignore
            embeddings_map["embedding_source"], task="embed_document", show_progress_bar=True
        )

        # For each feature, find the nodes that are > min_similarity similar
        merge_report = find_similar_nodes(
            min_similarity=min_similarity,
            **embeddings_map,  # type: ignore
        )

    if dry_run:
        # Return the merge report now with the unchanged graph
        return graph, merge_report

    # Merge nodes
    graph = _merge_nodes(graph, merge_report.nodes.tolist(), feature_delimiter=feature_delimiter)

    return graph, merge_report
