def _num_tokens_from_string(
    string: str, tokenizer
) -> int:
    """Return the number of tokens in a text string."""
    return len(tokenizer.tokenize(string))