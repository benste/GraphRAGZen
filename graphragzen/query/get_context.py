from collections import Counter
from operator import itemgetter
from typing import List

import networkx as nx
import pandas as pd
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from graphragzen.text_embedding.vector_databases import VectorDatabase


def semantic_similar_entities(
    embedding_model: BaseEmbedder,
    vector_db: VectorDatabase,
    query: str,
    k: int,
    entity_types: List[str] = ["node", "edge"],
    features_to_match: List[str] = ["description"],
    score_threshold: float = 0.0,
) -> List[dict]:
    """Get entities with the highest semantic similarity to the query.

    Args:
        embedding_model (BaseEmbedder): The embedding model used to compute the query vector.
        vector_db (VectorDatabase): The vector database to search for similar entities.
        query (str): The query string to find similar entities.
        k (int): Maximum number of entities (nodes and edges) to return.
        entity_types (List[str], optional): Which entities to search. Defaults to ['node', 'edge'].
        features_to_match (List[str], optional): Entity features to perform semantic similarity
            search on with regards to the query. Defaults to ["description"].
        score_threshold (float, optional): Exclude all vector search results with a score worse
            than this. Defaults to 0.0.

    Returns:
        List[dict]: A list of dictionaries containing 'entity_type', 'entity_name', 'score'.
    """

    query_vector = embedding_model.embed(query, task="embed_query")

    results = vector_db.search(
        query_vector,
        k,
        filters={"feature": features_to_match, "entity_type": entity_types},
        score_threshold=score_threshold,
    )

    return [result["metadata"] | {"score": result["score"]} for result in results[0]]


def extra_inside_edges(
    graph: nx.Graph,
    entities: List[dict],
    k: int,
    only_new_edges: bool = True,
) -> List[dict]:
    """Return top edges by weight where both nodes are in the specified entities.

    Args:
        graph (nx.Graph): The graph containing nodes and edges.
        entities (List[dict]): List of entities who's nodes to check for connected edges.
        k (int): Maximum number of edges to return.
        only_new_edges (bool, optional): If True, returns only edges not yet in the entities,
            thus output length might be less than k. Defaults to True.

    Returns:
        List[dict]: A list of dictionaries containing 'entity_type', 'entity_name'.
    """

    return _extra_edges(graph, entities, k, "inside", only_new_edges)


def extra_outside_edges(
    graph: nx.Graph,
    entities: List[dict],
    k: int,
    only_new_edges: bool = True,
) -> List[dict]:
    """Return top edges by weight where only one node is in the specified entities.

    Args:
        graph (nx.Graph): The graph containing nodes and edges.
        entities (List[dict]): List of entities who's nodes to check for connected edges.
        k (int): Maximum number of edges to return.
        only_new_edges (bool, optional): If True, returns only edges not yet in the entities,
            this output length might be less than k. Defaults to True.

    Returns:
        List[dict]: A list of dictionaries containing 'entity_type', 'entity_name'.
    """

    return _extra_edges(graph, entities, k, "outside", only_new_edges)


def _extra_edges(
    graph: nx.Graph,
    entities: List[dict],
    k: int,
    inside_outside: str,
    only_new_edges: bool,
) -> List[dict]:
    """Find top edges by weight, filtered by inside or outside criteria.

    Args:
        graph (nx.Graph): The graph containing nodes and edges.
        entities (List[dict]): List of entities who's nodes to check for connected edges.
        k (int): Maximum number of edges to return.
        inside_outside (str): 'inside' to filter edges where both nodes are in the entities,
            or 'outside' to filter edges where only one node is in the entities.
            If left empty there will be no filtering before finding the top k edges.
        only_new_edges (bool): If True, returns only edges not yet in the entities, thus
            output length might be less than k.

    Returns:
        List[dict]: A list of dictionaries containing 'entity_type', 'entity_name'.
    """

    if k <= 0:
        return []

    # Get all the edges coupled to each nodes in the entities
    node_names = []
    node_edges = []
    edges = []
    for entity in entities:
        if entity["entity_type"] == "node":
            node_edges += list(graph.edges(entity["entity_name"], data=True))
            node_names.append(entity["entity_name"])
        elif entity["entity_type"] == "edge":
            edges.append(entity["entity_name"])

    # Keep only the inside or outside edges
    if inside_outside == "inside":
        node_edges = [
            edge for edge in node_edges if edge[0] in node_names and edge[1] in node_names
        ]
    elif inside_outside == "outside":
        node_edges = [
            edge for edge in node_edges if not (edge[0] in node_names and edge[1] in node_names)
        ]

    if not node_edges:
        return []

    # Put edge information in a dataframe for easier manipulation and get the edge weight
    node_edges_df = pd.DataFrame(node_edges, columns=["node1", "node2", "metadata"])
    node_edges_df["entity_name"] = node_edges_df.apply(
        lambda edge: (edge.node1, edge.node2), axis=1
    )
    node_edges_df["weight"] = node_edges_df.metadata.apply(lambda m: float(m.get("weight", 0.0)))

    if only_new_edges:
        # Keep edges that are not yet in the entities
        node_edges_df = node_edges_df[
            node_edges_df.entity_name.apply(lambda edge: edge not in edges)
        ]

    # Make sure we don't have duplicate edges
    node_edges_df.drop_duplicates(subset="entity_name")

    # Return the top edges by weight
    top_edges = node_edges_df.sort_values(by="weight", ascending=False).iloc[:k]
    return [
        {"entity_type": "edge", "entity_name": entity_name} for entity_name in top_edges.entity_name
    ]


def source_texts(
    source_documents: pd.DataFrame,
    graph: nx.Graph,
    entities: List[dict],
    k: int,
    id_key: str = "chunk_id",
    source_key: str = "chunk",
    feature_delimiter: str = "\n",
) -> List[str]:
    """Find the top source texts by frequency related to the specified entities.

    Args:
        source_documents (pd.DataFrame): DataFrame containing the source documents with unique
            identifiers.
        graph (nx.Graph): A networkx graph containing nodes and edges.
        entities (List[dict]): A list of dictionaries representing entities, with each entity having
            'entity_type': A string that is either 'node' or 'edge'.
            'entity_name': A string representing the name of the entity.
        k (int): The number of top source texts to return, based on their frequency of occurrence.
        id_key (str): The column name in `source_documents` that represents unique document IDs.
            Defaults to "chunk_id".
        source_key (str): The column name in `source_documents` containing the source text. Defaults
            to "chunk".
        feature_delimiter (str): The delimiter used to split the "source_id" values in the Graph
            metadata. Default is newline.

    Returns:
        List[str]: A list of source texts corresponding to the top most frequent occurring sources.
    """

    # First get all the source id's
    nodes = []
    edges = []
    for entity in entities:
        if entity["entity_type"] == "node":
            nodes.append(entity["entity_name"])
        elif entity["entity_type"] == "edge":
            edges.append(entity["entity_name"])

    source_ids = []
    for entity in itemgetter(*nodes)(graph.nodes) + itemgetter(*edges)(graph.edges):
        source_ids += entity.get("source_id", None).split(feature_delimiter)
    source_ids = [id for id in source_ids if id]  # Get rid of None's

    # Now get the k most occuring ID's and return the source text
    top_ids = Counter(source_ids).most_common(k)
    top_ids = [id[0] for id in top_ids]
    return source_documents[source_documents[id_key].apply(lambda key: key in top_ids)][
        source_key
    ].tolist()


def cluster_summaries(
    graph: nx.Graph,
    cluster_report: pd.DataFrame,
    entities: List[dict],
    k: int,
) -> List[dict]:
    """Find the top cluster summaries by occurence, followed by rank, related to the specified
    entities.

    Args:
        graph (nx.Graph): A networkx graph containing nodes with metadata about cluster associations
        cluster_report (pd.DataFrame): DataFrame containing the cluster information, including
            summaries and ratings.
        entities (List[dict]): A list of dictionaries representing entities, with each entity having
            'entity_type': A string that is 'node'.
            'entity_name': A string representing the name of the entity.
        k (int): The number of top cluster summaries to return, based on frequency and rank.

    Returns:
        List[str]: A list of dictionaries, each containing the title and summary of a cluster.
    """

    # Get the clusters the entities are related to
    nodes = []
    for entity in entities:
        if entity["entity_type"] == "node":
            nodes.append(entity["entity_name"])

    clusters: List[str] = []
    for entity in itemgetter(*nodes)(graph.nodes):
        clusters += entity.get("cluster")  # type: ignore

    # Find the by number of times each cluster was found in a node
    clusters_count = Counter(clusters)
    cluster_candidates = pd.DataFrame(
        {"cluster": clusters_count.keys(), "count": clusters_count.values()}
    )

    # Add the rank of each cluster
    cluster_candidates = cluster_candidates.merge(cluster_report, on="cluster")
    cluster_candidates["rank"] = cluster_candidates.description.apply(
        lambda d: float(d.get("rating", 1.0))
    )

    # Sort, first on count followed by rank, and keep top k
    top_clusters = cluster_candidates.sort_values(by=["count", "rank"], ascending=False).iloc[:k]

    # Join title and summaries per cluster and return
    return top_clusters.description.apply(
        lambda d: {"title": d.get("title", ""), "summary": d.get("summary", "")}
    ).tolist()
