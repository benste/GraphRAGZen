from typing import Any


def _num_tokens_from_string(string: str, tokenizer: Any) -> int:
    """Return the number of tokens in a text string.

    Args:
        string (str): To find the number of tokens for
        tokenizer (_type_): Should have the method 'tokenize()'

    Returns:
        int: number of tokens
    """
    return len(tokenizer.tokenize(string))
