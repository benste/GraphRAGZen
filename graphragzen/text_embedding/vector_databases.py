import os
import shutil
import warnings
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue


class VectorDatabase(ABC):

    distance_measure: str
    vector_size: int

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the client to communicate to the vector DB backend of your choice"""
        pass

    @abstractmethod
    def add_vectors(self, vectors: List[dict]) -> None:
        """Add vectors to the database

        Args:
            vectors (List[dict]): Each dict containing {"uuid": ..., "vector": ...}. Each dict may
                also contain the key and values {"metadata": ...}, which will be store with the
                vector and retrieved upon search.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        score_threshold: float = 0.0,
        filters: dict = {},
    ) -> List[List[dict]]:
        """Similarity search for each of the query vectors

        Args:
            query_vectors (np.ndarray): Vectors for n queries, shaped (n x embedding_size)
            k (int): Max number of results to return per query vector.
            score_threshold (float, optional): Exclude all vector search results with a score worse
                than his. Defaults to 0.0
            filters (dict, optional): {"key": "value_it_should_have", "key2": "value_it .....

        Returns:
            List[List[dict]]: Per query List[dict] with each dict containing
                {"uuid": ..., "score": ..., "metadata": ...}
        """
        pass


class QdrantLocalVectorDatabase(VectorDatabase):
    def __init__(
        self,
        vector_size: Optional[int] = None,
        database_location: Optional[str] = None,
        overwrite_existing_db: bool = False,
        distance_measure: Literal["Cosine", "Euclid", "Dot", "Manhattan"] = "Cosine",
        on_disk: bool = False,
    ) -> None:
        """Create or load a local Qdrant vector database and a client for interaction.

        Args:
            vector_size (int, optional): Length of the vectors to store. If a new database is
                created this must be provided. If a database if loaded this will be read from that
                database and the value provided here ignored.
            database_location (str, optional): Location to load the DB from or store a new DB.
                If not provided a new database will be created in `qdrant/databases/`.
                Defaults to None.
            overwrite_existing_db (str, optional): If True and a database is found at
                `database_location` it will be overwritten by a new database, otherwise the database
                found at `database_location` will be loaded. Defaults to False.
            distance_measure (Literal['Cosine', 'Euclid', 'Dot', 'Manhattan'], optional): Method to
                calculate distances between vectors. Defaults to 'Cosine'
            on_disk (bool, optional): If true, vectors are served from disk, improving RAM usage at
                the cost of latency. Defaults to False.
        """

        # If no db location is provided, set a temporary location
        if database_location is None:
            base_location = "qdrant/databases"
            location_postfix = 0
            database_location = f"{base_location}{location_postfix}"
            while os.path.exists(database_location):
                location_postfix += 1
                database_location = f"{base_location}{location_postfix}"

            warnings.warn(
                f"No location provided for vector database, creating a new database in {database_location}"  # noqa: E501
            )

        # Check if there's already a database at `database_location` and if we should overwrite it
        if os.path.exists(database_location) and overwrite_existing_db:
            # Remove the database already there
            shutil.rmtree(os.path.join(database_location, "collection"), ignore_errors=True)
            os.remove(os.path.join(database_location, ".lock"))
            os.remove(os.path.join(database_location, "meta.json"))

        # Create the client
        self.client = QdrantClient(path=database_location)

        # Save some variables internally
        self.vector_size = vector_size  # type: ignore
        self.distance_measure = distance_measure
        self.database_location = database_location

        # Qdrant databases need a collection
        self.collection_name = "default"
        self._add_collection_to_db(on_disk)

        # Make sure vector size is set correct
        self.client.get_collection("default").config.params.vectors.size  # type: ignore # noqa:E501

    def add_vectors(self, vectors: List[dict]) -> None:
        """Add vectors to the database

        Args:
            vectors (List[dict]): Each dict containing {"uuid": ..., "vector": ...}. Each dict may
                also contain the key and values {"metadata": ...}, which will be store with the
                vector and retrieved upon search.
        """

        # Prepare dicts for Qdrant PointStructs
        qdrant_compatible_vectors = [
            {
                "id": vector["uuid"],
                "vector": vector["vector"],
                "payload": vector.get("metadata", {}),
            }
            for vector in vectors
        ]

        # Add to client
        points = [PointStruct(**vector) for vector in qdrant_compatible_vectors]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        score_threshold: float = 0.0,
        filters: dict = {},
    ) -> List[List[dict]]:
        """Similarity search for each of the query vectors

        Args:
            query_vectors (np.ndarray): Vectors for n queries, shaped (n x embedding_size)
            k (int): Max number of results to return per query vector.
            score_threshold (float, optional): Exclude all vector search results with a score worse
                than his. Defaults to 0.0
            filters (dict, optional): {"key": "value_it_should_have", "key2": "value_it .....

        Returns:
            List[List[dict]]: Per query List[dict] with each dict containing
                {"uuid": ..., "score": ..., "metadata": ...}
        """

        # Format the filters for qdrant
        query_filter_list = []
        for key, value in filters.items():
            if isinstance(value, list):
                query_filter_list.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                query_filter_list.append(FieldCondition(key=key, match=MatchValue(value=value)))

        query_filter = Filter(must=query_filter_list)  # type: ignore

        results = []
        for vector in query_vectors:
            similar = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=k,
                score_threshold=score_threshold,
            )
            # Format normalized according to the VectorDatabase class
            results.append(
                [
                    {
                        "uuid": r.id,
                        "score": r.score,
                        "metadata": r.payload,
                    }
                    for r in similar
                ]
            )

        return results

    def save(self, location: str) -> None:
        """This is specific to the Qdrant local VectorDatabae. This function just copies the DB to a
        new location. Does not work for :memory: qdrant client instances since they life in RAM.

        Args:
            location (str): path to store the DB
        """

        if location:
            if os.path.exists(location):
                # Database already exists at this location
                raise Exception(f"Trying to save vector DB to {location}, but path already exists.")

            shutil.copytree(src=self.client._client.location, dst=location)  # type: ignore

    def _add_collection_to_db(
        self,
        on_disk: bool = False,
    ) -> None:
        """Initiate a new collection in the DB.

        Args:
            on_disk (bool, optional): If true, vectors are served from disk, improving RAM usage at
                the cost of latency. Defaults to False.
        """

        if self.client.collection_exists(self.collection_name):
            return None

        vectors_config = VectorParams(
            size=self.vector_size,
            distance=self.distance_measure,  # type: ignore
            on_disk=on_disk,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )
