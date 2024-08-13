import os
from collections import defaultdict

import pandas as pd

from graphragzen.llm.base_llm import LLM
from graphragzen.preprocessing import utils
from graphragzen.typing import preprocessing


def raw_documents(**kwargs: preprocessing.RawDocumentsConfig) -> pd.DataFrame:
    """loads files from folder path and subfolders.

    Kwargs:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's
            content. Defaults to 'content'.
    Returns:
        pd.DataFrame: Includes the columns 'document_path' and 'document_id'
    """
    config = preprocessing.RawDocumentsConfig(**kwargs)

    # Walk the folder path, find text files and load them
    folder_path = config.raw_documents_folder
    df = defaultdict(list)
    file_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                df["document_path"].append(os.path.join(root, file))
                df[config.raw_content_column].append(open(df["document_path"][-1], "r").read())
                df["document_id"].append(str(file_id))
                file_id += 1

    return pd.DataFrame(df)


def chunk_documents(
    dataframe: pd.DataFrame, llm: LLM, **kwargs: preprocessing.ChunkConfig
) -> pd.DataFrame:
    """Chunk documents based on number of tokens

    Args:
        dataframe (pd.DataFrame): Containing the documents to chunk
        llm (LLM)

    Kwargs:
        column_to_chunk (str, optional): Column to chunk, Defaults to 'content'.
        results_column (str, optional): Column to write chunks to, Defaults to 'chunk'.
        id_column (str, optional): Column with which to later refence the source chunk.
            Defaults to 'chunk_id'.
        window_size (str, optional): Number of tokens in each chunk, Defaults to 300.
        overlap (str, optional): Number of tokens chunks overlap, Defaults to 100.

    Returns:
        pd.DataFrame: All columns in the input dataframe are exploded with the chunks
            allowing referencing
    """
    config = preprocessing.ChunkConfig(**kwargs)

    results_column = config.results_column
    len_column = config.results_column + "_len"
    id_column = config.results_column + "_id"

    # Apply chunking per document, also saving the number of tokens in each chunk
    dataframe[results_column], dataframe[len_column] = zip(
        *dataframe[config.column_to_chunk].apply(
            lambda c: utils.chunk_tokens(c, llm, config.window_size, config.overlap)
        )
    )

    # Map each chunk back to the correct row
    dataframe = dataframe.explode([results_column, len_column])

    # Give each chunk a unique ID
    dataframe[id_column] = list(range(len(dataframe)))

    # TODO: drop content column to save space?

    return dataframe
