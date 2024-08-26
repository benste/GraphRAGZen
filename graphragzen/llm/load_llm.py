from typing import Any, Union

from graphragzen.llm.gemma2 import Gemma2GGUF

from .typing import LlmLoadingConfig


def load_gemma2_gguf(**kwargs: Union[dict, LlmLoadingConfig, Any]) -> Gemma2GGUF:
    """Load gguf version of Gemma 2

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.

    Returns:
        Gemma2GGUF: see `graphragzen.llm.gemma2.Gemma2GGUF`
    """
    config = LlmLoadingConfig(**kwargs)  # type: ignore

    return Gemma2GGUF(config=config)


def load_gemma2_huggingface() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass


def load_openAI() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass
