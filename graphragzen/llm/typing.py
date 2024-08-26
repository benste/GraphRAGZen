from pydantic import ConfigDict

from ..typing.MappedBaseModel import MappedBaseModel


class ChatNames(MappedBaseModel):
    """The ChatNames the LLM expects in the prompt

    Args:
        user (str, optional): Name of the user. Defaults to 'user'.
        model (str, optional): Name of the model Defaults to 'assistant'.
    """

    user: str = "user"
    model: str = "assistant"


class LlmLoadingConfig(MappedBaseModel):
    """Config for loading local LLM

    Args:
        model_storage_path (str): Path to the model on the local filesystem
        tokenizer_URI (str): Huggingface URI for the tokenizer
        context_size (int, optional): Size of the context window in tokens. Defaults to 8192
        use_cache (bool, optional): Use a cache to find output for previously processed inputs in
            stead of re-generating output from the input. Default to True.
        cache_persistent (bool, optional): Append the cache to a file on disk so it can be re-used
            between runs. If False will use only in-memory cache. Default to True
        persistent_cache_file (str, optional): The file to store the persistent cache.
            Defaults to './llm_persistent_cache.yaml'.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_storage_path: str
    tokenizer_URI: str
    context_size: int = 8192
    use_cache: bool = True
    cache_persistent: bool = True
    persistent_cache_file: str = "./llm_persistent_cache.yaml"
