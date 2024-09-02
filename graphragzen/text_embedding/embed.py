from typing import List, Optional, Union
from uuid import uuid4

import networkx as nx
import pandas as pd
from graphragzen.text_embedding.embedding_models import BaseEmbedder
from graphragzen.text_embedding.vector_db import add_points_to_vector_db
from qdrant_client import QdrantClient


def embed_graph_features(
    graph: nx.Graph,
    embedding_model: BaseEmbedder,
    features_to_embed: Union[List[str], str],
    vector_db_client: Optional[QdrantClient] = None,
) -> pd.DataFrame:
    """Text embed features of entities from a graph.

    Args:
        graph (nx.Graph):
        embedding_model (BaseEmbedder):
        features_to_embed (List[str]): Features of the entities (node or edge) the embed
        vector_db_client (QdrantClient, optional): If provided, will add the embedding to the
            vector database.

    Returns:
        pd.DataFrame: with keys 'entity_name', 'entity_type', 'feature', 'uuid', 'vector'
    """

    if isinstance(features_to_embed, str):
        features_to_embed = [features_to_embed]

    embeddings = []
    for feature_to_embed in features_to_embed:
        # Get the node features to embed
        for entity in graph.nodes(data=True):
            entity_name = entity[0]
            entity_features = entity[1]
            if feature_to_embed in entity_features:
                embeddings.append(
                    {
                        "entity_name": entity_name,
                        "entity_type": "node",
                        "feature": feature_to_embed,
                        "uuid": str(uuid4()),
                        "to_embed": entity_features[feature_to_embed],
                    }
                )

        # Get the edge features to embed
        for entity in graph.edges(data=True):
            entity_name = (entity[0], entity[1])
            entity_features = entity[2]
            if feature_to_embed in entity_features:
                embeddings.append(
                    {
                        "entity_type": "edge",
                        "entity_name": entity_name,
                        "feature": feature_to_embed,
                        "uuid": str(uuid4()),
                        "to_embed": entity_features[feature_to_embed],
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
    if vector_db_client:
        add_points_to_vector_db(vector_db_client, embeddings_df)

    return embeddings_df


def embed_dataframe(
    dataframe: pd.DataFrame,
    embedding_model: BaseEmbedder,
    vector_db_client: Optional[QdrantClient] = None,
    columns_to_embed: List[str] = [],
) -> pd.DataFrame:
    """_summary_

    Args:
        dataframe (pd.DataFrame)
        embedding_model (BaseEmbedder)
        vector_db_client (QdrantClient, optional): If provided, will add the embedding to the
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
        if vector_db_client:
            dataframe["uuid"] = [str(uuid4()) for _ in range(len(dataframe))]
            add_points_to_vector_db(vector_db_client, dataframe)
            dataframe.rename(columns={"uuid", f"{to_embed}_uuid"}, inplace=True)

    return dataframe
