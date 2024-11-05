from typing import List, Optional
from urllib.parse import urljoin

import requests
import tiktoken


class ApiTokenizer:
    """tokenizes and detokenizes using API endpoints"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def tokenize(self, content: str, base_url: Optional[str] = None) -> List[int]:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, "tokenize"), json={"content": content})
        return result.json().get("tokens")

    def convert_tokens_to_string(self, tokens: List[int], base_url: Optional[str] = None) -> str:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, "detokenize"), json={"tokens": tokens})
        return result.json().get("content")


class TikTokenTokenizer:
    """tokenizes and detokenizes using tiktoken library"""

    def __init__(self, model_name: str):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding(model_name)

    def tokenize(self, content: str) -> List[int]:
        return self.encoding.encode(content)

    def convert_tokens_to_string(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)
