from typing import Optional

from pydantic import BaseModel
    
class RawDocumentsConfig(BaseModel):
    """Config for raw document loading

    Args:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's content.
            Defaults to 'content'.
    """
    raw_documents_folder: str
    raw_content_column: Optional[str] = 'content'
    
class ChunkConfig(BaseModel):
    """Config for chunking documents

    Args:
        column_to_chunk (str, optional): Column to chunk, Defaults to 'content'.
        results_column (str, optional): Column to write chunks to, Defaults to 'chunk'.
        id_column (str, optional): column with which to later refence the source text, Defaults to 'chunk_id'.
        window_size (str, optional): Number of tokens in each chunk, Defaults to 300.
        overlap (str, optional): Number of tokens chunks overlap, Defaults to 100.
    """
    column_to_chunk: Optional[str] = 'content'
    results_column: Optional[str] = 'chunk'
    id_column: Optional[str] = 'chunk_id'
    window_size: Optional[int] = 300
    overlap: Optional[int] = 100
    
class PreprocessConfig(BaseModel):
    """Config for preprocessing raw documents

    Args:
        BaseModel (RawDocumentsConfig)
        chunk (ChunkConfig)
    """
    raw_documents: RawDocumentsConfig
    chunk: ChunkConfig
    
