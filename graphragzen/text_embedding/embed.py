from typing import List, Optional, Union
from uuid import uuid4

import networkx as nx
import pandas as pd
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from graphragzen.text_embedding.vector_databases import VectorDatabase


def embed_graph_features(
    graph: nx.Graph,
    embedding_model: BaseEmbedder,
    features_to_embed: Union[List[str], str],
    entities_to_embed: Union[List[str], str] = ["edge", "node"],
    vector_db: Optional[VectorDatabase] = None,
) -> pd.DataFrame:
    """Text embed features of entities from a graph.

    Args:
        graph (nx.Graph):
        embedding_model (BaseEmbedder):
        features_to_embed (List[str]): Features of the entities the embed.
        entities_to_embed (List[str], optional): Which type of entities (node or edge) to look for
        the features to embed. Defaults to ['edge', 'node'].
        vector_db (VectorDatabase, optional): If provided, will add the embedding to the
            vector database.

    Returns:
        pd.DataFrame: with keys 'entity_name', 'entity_type', 'feature', 'uuid', 'vector'
    """

    if isinstance(features_to_embed, str):
        features_to_embed = [features_to_embed]

    embeddings = []
    for feature_to_embed in features_to_embed:
        # Get the node features to embed
        entity_names = []
        entity_features = []
        entity_types = []
        if "node" in entities_to_embed:
            for entity in graph.nodes(data=True):
                entity_names.append(entity[0])
                entity_features.append(entity[1])
                entity_types.append("node")

        if "edge" in entities_to_embed:
            for entity in graph.edges(data=True):
                entity_names.append((entity[0], entity[1]))
                entity_features.append(entity[2])
                entity_types.append("edge")

        for name, features, type in zip(entity_names, entity_features, entity_types):
            if feature_to_embed in features:
                embeddings.append(
                    {
                        "entity_name": name,
                        "entity_type": type,
                        "feature": feature_to_embed,
                        "uuid": str(uuid4()),
                        "to_embed": features[feature_to_embed],
                    }
                )

    # Convert to dataframe, embed, and add the embedding to the dataframe
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df["vector"] = embedding_model.embed(
        embeddings_df.to_embed, task="embed_document"
    ).tolist()
    embeddings_df.drop(columns="to_embed", inplace=True)  # don't need to store this

    # If vector database is provided, add the vectors to it
    embeddings_df["metadata"] = (
        embeddings_df[["entity_type", "entity_name", "feature"]].T.to_dict().values()
    )
    if vector_db:
        vector_db.add_vectors(embeddings_df.to_dict(orient="records"))

    return embeddings_df


def embed_dataframe(
    dataframe: pd.DataFrame,
    embedding_model: BaseEmbedder,
    vector_db: Optional[VectorDatabase] = None,
    columns_to_embed: List[str] = [],
) -> pd.DataFrame:
    """Embed specific columns of a database, and add each embedding to the vector DB

    Args:
        dataframe (pd.DataFrame)
        embedding_model (BaseEmbedder)
        vector_db_client (VectorDatabase, optional): If provided, will add the embedding to the
            vector database.
        columns_to_embed (List[str], optional): Which columns to embed. If not provided embeds all
            columns of that contain strings or Null. Defaults to [].

    Returns:
        pd.DataFrame: With vector columns added as f"{original_column}_vector"
    """

    if not columns_to_embed:
        # Get all columns that contain strings or Null
        for column in dataframe:
            if pd.api.types.is_string_dtype(dataframe[column].dropna()):
                columns_to_embed.append(column)

    for to_embed in columns_to_embed:
        # Make a placehold for Na's, we'll replace their vectors with nan's later
        isna = dataframe[to_embed].isna().tolist()
        dataframe[to_embed][isna] = "_na_placeholder_"

        # Get vectors
        vectors = embedding_model.embed(dataframe[to_embed])

        # Make vectors of nan's for the inputs that were originally None
        vectors[isna] = None

        # Write back to df
        dataframe[f"{to_embed}_vector"] = vectors.tolist()

        # If vector database is provided, add the vectors to it
        if vector_db:
            dataframe[f"{to_embed}_uuid"] = [str(uuid4()) for _ in range(len(dataframe))]

            for_vector_db = dataframe[[f"{to_embed}_vector", f"{to_embed}_uuid"]].rename(
                columns={f"{to_embed}_vector": "vector", f"{to_embed}_uuid": "uuid"}
            )
            vector_db.add_vectors(for_vector_db.to_dict(orient="records"))

    return dataframe
