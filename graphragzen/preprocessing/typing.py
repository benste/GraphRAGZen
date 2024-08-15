from typing import Optional

from ..typing.MappedBaseModel import MappedBaseModel


class LoadTextDocumentsConfig(MappedBaseModel):
    """Config for text document loading

    Args:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's
            content. Defaults to 'content'.
    """

    raw_documents_folder: str
    raw_content_column: Optional[str] = "content"


class ChunkConfig(MappedBaseModel):
    """Config for chunking documents

    Args:
        column_to_chunk (str, optional): Column to chunk, Defaults to 'content'.
        results_column (str, optional): Column to write chunks to, Defaults to 'chunk'.
        id_column (str, optional): Column with which to later refence the source chunk.
            Defaults to 'chunk_id'.
        window_size (str, optional): Number of tokens in each chunk, Defaults to 300.
        overlap (str, optional): Number of tokens chunks overlap, Defaults to 100.
        method (str, optional): What to chunk. Currently only 'tokens' is implemented meaning
            column_to_chunk is first tokenized and the tokens are chunked.
    """

    column_to_chunk: str = "content"
    results_column: str = "chunk"
    id_column: str = "chunk_id"
    window_size: int = 300
    overlap: int = 100
    method: str = "tokens"
