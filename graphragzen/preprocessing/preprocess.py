import os
from collections import defaultdict
import re
from typing import Sequence, Any

import html
import pandas as pd

from graphragzen.llm.base_llm import LLM
from .typing import LoadTextDocumentsConfig, ChunkConfig


def load_text_documents(**kwargs: Any) -> pd.DataFrame:
    """loads files from folder path and subfolders.

    Kwargs:
        raw_documents_folder (str): Folder to search for text documents
        raw_content_column (str, optional): Name of the dataframe column to store each document's
            content. Defaults to 'content'.
    Returns:
        pd.DataFrame: Includes the columns 'document_path' and 'document_id'
    """
    config = LoadTextDocumentsConfig(**kwargs)  # type: ignore

    # Walk the folder path, find text files and load them
    folder_path = config.raw_documents_folder
    df = defaultdict(list)
    file_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                df["document_path"].append(os.path.join(root, file))
                df[config.raw_content_column].append(open(df["document_path"][-1], "r").read())  # type: ignore  # noqa: E501
                df["document_id"].append(str(file_id))
                file_id += 1

    return pd.DataFrame(df)


def chunk_documents(dataframe: pd.DataFrame, llm: LLM, **kwargs: Any) -> pd.DataFrame:
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
    config = ChunkConfig(**kwargs)  # type: ignore

    results_column: str = config.results_column
    len_column: str = results_column + "_len"
    id_column: str = results_column + "_id"

    # Method of chunking
    if config.method == "tokens":
        to_chunk = dataframe[config.column_to_chunk].apply(llm.tokenize)

    # Apply chunking per document, also saving the length of each chunk
    dataframe[results_column], dataframe[len_column] = zip(
        *to_chunk.apply(lambda c: chunk(c, **config))
    )

    # Map each chunk back to the correct row
    dataframe = dataframe.explode([results_column, len_column])

    # 'untokenize' if required
    if config.method == "tokens":
        dataframe[results_column] = dataframe[results_column].apply(llm.untokenize)

    # Give each chunk a unique ID
    dataframe[id_column] = list(range(len(dataframe)))

    # TODO: drop content column to save space?

    return dataframe


def chunk(inp: Sequence, **kwargs: Any) -> tuple[list, list]:
    """Chunk an iterable using a sliding window

    Args:
        inp (Iterable): Iterable to chunk

    Kwargs:
        window_size (int): size of the chunk window
        overlap (int): overlap between windows

    Returns:
        tuple[list, list]: (chunks, chunk lengths)
    """
    config = ChunkConfig(**kwargs)  # type: ignore

    chunks = []
    chunk_lengths = []
    for start_index in range(0, len(inp), config.window_size - config.overlap):
        chunk = inp[start_index : start_index + config.window_size]
        chunks.append(chunk)
        chunk_lengths.append(len(chunk))

    return chunks, chunk_lengths


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted
    characters.
    """
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    result = result.lstrip('"').rstrip('"')
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python  # noqa: E501
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
