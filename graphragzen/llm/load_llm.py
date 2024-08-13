from graphragzen.llm.gemma2 import Gemma2GGUF
from graphragzen.typing import LlmLoadingConfig


def load_gemma2_gguf(**kwargs: LlmLoadingConfig) -> Gemma2GGUF:
    """Load gguf version of Gemma 2

    Kwargs:
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
    # TODO: implement
    pass


def load_openAI() -> None:
    # TODO: implement
    pass
