import os
import shutil
import warnings
from typing import Any, Union, List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .typing import VectorDBConfig


def create_vector_db(**kwargs: Union[dict, VectorDBConfig, Any]) -> QdrantClient:
    """Create an instance of a vector database and return a client for interaction.

    Args:
        vector_size (int): Length of the vectors to store .
        database_location (str, optional): Location to store the DB. If not provided
            `create_vector_db` will create a temporary location and update this variable.
        overwrite_existing_db (str, optional): If True and a database is found at
            `database_location` it will be overwritten by a new database.
            If False and a database is found at `database_location` an exception is raised.
            Defaults to False.
        distance_measure (Literal['Cosine', 'Euclid', 'Dot', 'Manhattan'], optional): Method to
            calculate distances between vectors. Defaults to 'Cosine'
        on_disk (bool, optional): If true, v`ectors are served from disk, improving RAM usage at the
            cost of latency. Defaults to False.
        collection_name (str, optional): QDrant can have multiple separated collections in one DB,
            each collection containing their own vectors. For the purpose of RAG it's recommended to
            create a new DB for each new project and stick to one collection name per DB.

    Returns:
        QdrantClient
    """
    config = VectorDBConfig(**kwargs)  # type: ignore

    # If no db location is provided, set a temporary location
    if config.database_location is None:
        base_location = "qdrant/temp_db"
        location_postfix = 0
        config.database_location = f"{base_location}{location_postfix}"
        while os.path.exists(config.database_location):
            location_postfix += 1
            config.database_location = f"{base_location}{location_postfix}"

        warnings.warn(
            f"No location provided for vector DB, creating temporary db in {config.database_location}"  # noqa: E501
        )

    # Check if there's already a database at `database_location`
    if os.path.exists(config.database_location):
        if config.overwrite_existing_db:
            # Remove the database already there
            shutil.rmtree(os.path.join(config.database_location, "collection"), ignore_errors=True)
            os.remove(os.path.join(config.database_location, ".lock"))
            os.remove(os.path.join(config.database_location, "meta.json"))
        else:
            # Database already exists and we do not have permission to overwrite
            raise Exception(
                f"""{config.database_location} already exists. Set 'overwrite_db' to True or load
                the existing database using `load_vector_db`"""
            )

    # Create the client
    client = QdrantClient(path=config.database_location)

    # Create a collection
    add_collection_to_db(client=client, config=config)

    return client


def save_vector_db(client: QdrantClient, **kwargs: Union[dict, VectorDBConfig, Any]) -> None:
    """This just copies the DB to a new location. Does not work for :memory: qdrant client instances
    since they life in RAM.

    Args:
        client (QdrantClient)
        database_location (str): path to store the DB
    """

    config = VectorDBConfig(**kwargs)  # type: ignore

    if config.database_location:
        if os.path.exists(config.database_location):
            # Database already exists at this location
            raise Exception(
                f"Trying to save vector DB to {config.database_location}, but path already exists."
            )

        shutil.copytree(src=client._client.location, dst=config.database_location)  # type: ignore


def load_vector_db(**kwargs: Union[dict, VectorDBConfig, Any]) -> QdrantClient:
    config = VectorDBConfig(**kwargs)  # type: ignore
    return QdrantClient(path=config.database_location)


def add_collection_to_db(client: QdrantClient, **kwargs: Union[dict, VectorDBConfig, Any]) -> None:
    config = VectorDBConfig(**kwargs)  # type: ignore

    if client.collection_exists(config.collection_name):
        warnings.warn(
            f"collection {config.collection_name} already exists in vector DB, won't add it."
        )
        return None

    vectors_config = VectorParams(
        size=config.vector_size,
        distance=config.distance_measure,  # type: ignore
        on_disk=config.on_disk,
    )
    client.create_collection(collection_name=config.collection_name, vectors_config=vectors_config)


def add_points_to_vector_db(
    client: QdrantClient, vector_df: pd.DataFrame, collection_name: str = "default"
) -> None:
    """Add vectors with their ID and metadata to the vector DB

    Args:
        client (QdrantClient)
        vector_df (pd.DataFrame): Expected to contain the columns
            - uuid (List[str])
            - vector (Union[List[int], np.ndarray])
            With optionally
            - metadata (dict)
        collection_name (str, optional): Collection to add the vectors to.
            QDrant can have multiple separated collections in one DB,
            each collection containing their own vectors. For the purpose of RAG it's recommended to
            create a new DB for each new project and stick to one collection name per DB.

    The metadata added to the vectors in the DB can later be used as a filter when querying.
    """

    # Make sure required columns are in DF
    required = ["uuid", "vector"]
    if any(key not in vector_df for key in required):
        raise Exception(
            "Trying to add vector to DB, but vector_df is missing one of 'uuid' or 'vector' column"
        )

    # If metadata is not in the dataframe, make a dummy column
    if "metadata" not in vector_df:
        vector_df["metadata"] = {}

    points = [
        PointStruct(id=uuid, vector=vector, payload=payload)
        for uuid, vector, payload in zip(
            vector_df.uuid.tolist(), vector_df.vector.tolist(), vector_df.metadata.tolist()
        )
    ]
    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    
    
def search_vector_db(
    client: QdrantClient,
    query_vector: np.ndarray,
    k: int,
    filters: Optional[List[dict]] = {},
    collection_name: str = "default",
    ) -> Any:
    """Search the vector database with optional filters on the metadata attached to the points.
    
    Note: The distance measure used to search is set when the database is created, see 
    `graphragzen.text_embedding.vector_db.create_vector_db()`

    Args:
        client (QdrantClient)
        query_vector (np.ndarray): Shaped (n,) or (n,1) where n is the size of the vectors to search
        k (int): Max number of results to return
        filters (List[dict]), optional: [{"key": "value_it_should_have"}, {"key": "value_it .....
        collection_name (str, optional): Collection to search vectors in.
            QDrant can have multiple separated collections in one DB,
            each collection containing their own vectors. For the purpose of RAG it's recommended to
            create a new DB for each new project and stick to one collection name per DB.


    Returns:
        Any: _description_
    """
    
    # Format the filters for qdrant
    query_filter = []
    for key, value in filters.items():
        query_filter.append([FieldCondition(key=key, match=MatchValue(value=value))])
    query_filter = Filter(must=query_filter)
    
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=k
    )