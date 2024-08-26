from typing import List, Literal, Optional

from pydantic import ConfigDict

from ..typing.MappedBaseModel import MappedBaseModel


class EmbedderLoadingConfig(MappedBaseModel):
    """Config for loading embedding model

    note: Either or both `model_storage_path` or `huggingface_URI` must be provided. When both are
    provided `model_storage_path` takes precedence.

    Args:
        model_storage_path (str, optional): Path to the model on the local filesystem
        huggingface_URI (str, optional): Huggingface URI of the model.
            Defaults to "nomic-ai/nomic-embed-text-v1.5".
    """

    model_config = ConfigDict(protected_namespaces=())

    model_storage_path: Optional[str] = None
    huggingface_URI: str = "nomic-ai/nomic-embed-text-v1.5"


class VectorDBConfig(MappedBaseModel):
    """Config for the vector DB

    Args:
        database_location (str, optional): Location to store the DB. If not provided
            `create_vector_db` will create a temporary location and update this variable.
        overwrite_existing_db (str, optional): If True and a database is found at
            `database_location` it will be overwritten by a new database.
            If False and a database is found at `database_location` an exception is raised.
            Defaults to False.
        vector_size (int): Length of the vectors to store .
        distance_measure (Literal['Cosine', 'Euclid', 'Dot', 'Manhattan'], optional): Method to
            calculate distances between vectors. Defaults to 'Cosine'
        on_disk (bool, optional): If true, v`ectors are served from disk, improving RAM usage at the
            cost of latency. Defaults to False.
        collection_name (str, optional): QDrant can have multiple separated collections in one DB,
            each collection containing their own vectors. For the purpose of RAG it's recommended to
            create a new DB for each new project and stick to one collection name per DB.
    """

    database_location: Optional[str] = None
    overwrite_existing_db: bool = False
    vector_size: int
    distance_measure: Literal["Cosine", "Euclid", "Dot", "Manhattan"] = "Cosine"
    on_disk: bool = False
    collection_name: str = "default"


class EmbedGraphFeaturesConfig(MappedBaseModel):
    """Config for graph feature text embeddings

    Args:
        features_to_embed (List[str]): Which features to embed
    """

    features_to_embed: List[str]


class EmbedDataframeConfig(MappedBaseModel):
    """Config for dataframe column text embeddings

    Args:
        columns_to_embed (List[str], optional): Which columns to embed. If not provided embeds all
            columns of that contain strings or Null. Defaults to [].
    """

    columns_to_embed: List[str] = []
