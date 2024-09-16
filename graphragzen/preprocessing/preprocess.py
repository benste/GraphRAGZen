import html
import re
from copy import deepcopy
from typing import Any, List, Sequence, Union

import pandas as pd
from graphragzen.llm.base_llm import LLM


def chunk_documents(
    input: Union[pd.DataFrame, List[str]],
    llm: LLM,
    column_to_chunk: str = "content",
    results_column: str = "chunk",
    id_column: str = "chunk_id",
    window_size: int = 300,
    overlap: int = 100,
    method: str = "tokens",
) -> pd.DataFrame:
    """Chunk documents based on number of tokens

    Args:
        input (Union[pd.DataFrame, List[str]]): Containing the documents to chunk
        llm (LLM):
        column_to_chunk (str, optional): Column to chunk. Defaults to 'content'.
        results_column (str, optional): Column to write chunks to, Defaults to 'chunk'.
        id_column (str, optional): Column to write chunk ID's to. Can later be used to refence the
            source chunk. Defaults to 'chunk_id'.
        window_size (str, optional): Number of tokens in each chunk, Defaults to 300.
        overlap (str, optional): Number of tokens chunks overlap, Defaults to 100.
        method (str, optional): What to chunk. Currently only 'tokens' is implemented meaning
            column_to_chunk is first tokenized and the tokens are chunked.

    Returns:
        pd.DataFrame: All columns in the input dataframe are exploded with the chunks
        allowing referencing
    """

    if isinstance(input, list):
        chunked_df = pd.DataFrame({column_to_chunk: input})
    else:
        chunked_df = deepcopy(input)

    len_column: str = results_column + "_len"

    # Method of chunking
    if method == "tokens":
        to_chunk = chunked_df[column_to_chunk].apply(llm.tokenize)

    # Apply chunking per document, also saving the length of each chunk
    chunked_df[results_column], chunked_df[len_column] = zip(
        *to_chunk.apply(lambda c: chunk(c, window_size, overlap))
    )

    # Map each chunk back to the correct row
    chunked_df = chunked_df.explode([results_column, len_column])

    # 'untokenize' if required
    if method == "tokens":
        chunked_df[results_column] = chunked_df[results_column].apply(llm.untokenize)

    # Give each chunk a unique ID
    chunked_df[id_column] = [str(id) for id in range(len(chunked_df))]

    # TODO: drop content column to save space?

    return chunked_df


def chunk(
    inp: Sequence,
    window_size: int = 300,
    overlap: int = 100,
) -> tuple[list, list]:
    """Chunk an sequence using a sliding window

    Args:
        inp (Iterable): Iterable to chunk
        window_size (int, optional): size of the chunk window. Defaults to 300.
        overlap (int, optional): overlap between windows. Defaults to 100.

    Returns:
        tuple[list, list]: (chunks, chunk lengths)
    """

    chunks = []
    chunk_lengths = []
    for start_index in range(0, len(inp), window_size - overlap):
        chunk = inp[start_index : start_index + window_size]
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
