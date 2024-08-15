from typing import Any

from graphragzen.llm.gemma2 import Gemma2GGUF

from .typing import LlmLoadingConfig


def load_gemma2_gguf(**kwargs: Any) -> Gemma2GGUF:
    """Load gguf version of Gemma 2

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192

    Returns:
        Gemma2GGUF: see `graphragzen.llm.gemma2.Gemma2GGUF`
    """
    config = LlmLoadingConfig(**kwargs)  # type: ignore

    return Gemma2GGUF(
        model_path=config.model_storage_path,
        tokenizer_URI=config.tokenizer_URI,
        context_size=config.context_size,
    )


def load_gemma2_huggingface() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass


def load_openAI() -> None:
    """NOT YET IMPLEMENTED"""
    # TODO: implement
    pass
