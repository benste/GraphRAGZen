from typing import Any
import re

import html
from graphragzen.llm.base_llm import LLM

def chunk_tokens(text: str, llm: LLM, window_size: int, overlap: int) -> tuple[list]:
    """Tokenize a text and chunk it using a sliding window

    Args:
        text (str): to tokenize and subsequently chunk
        llm (LLM): should have the method `LLM.tokenize` defined
        window_size (int): size of the chunk window
        overlap (int): overlap between windows

    Returns:
        list: Actual chunks
    """
    
    tokenized = llm.tokenize(text)
    
    chunks = []
    chunks_length = []
    for start_index in range(0, len(tokenized), window_size - overlap):
        chunk = tokenized[start_index:start_index+window_size]
        chunks.append(llm.untokenize(chunk))
        chunks_length.append(len(chunk))
        
    return chunks, chunks_length

def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    result = result.lstrip('"').rstrip('"')
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    
    
    
    